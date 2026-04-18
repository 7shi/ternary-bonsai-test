import argparse
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper
import torch


FP8_TYPE_MAP = {
    "e4m3fn": (TensorProto.FLOAT8E4M3FN, torch.float8_e4m3fn),
    "e5m2": (TensorProto.FLOAT8E5M2, torch.float8_e5m2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-model-path",
        default="model_q2/Ternary-Bonsai-8B-ONNX/onnx/model_q2f16.onnx",
        help="Path to the source q2 ONNX model.",
    )
    parser.add_argument(
        "--output-model-path",
        default="model_q2_to_fp8/model.onnx",
        help="Path to write the converted dense fp8 ONNX model.",
    )
    parser.add_argument(
        "--fp8-format",
        choices=sorted(FP8_TYPE_MAP),
        default="e4m3fn",
        help="FP8 storage format to use for converted weights.",
    )
    return parser.parse_args()


def copy_metadata_files(source_root: Path, output_root: Path) -> None:
    for file_name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
    ]:
        source_path = source_root / file_name
        if source_path.exists():
            shutil.copy2(source_path, output_root / file_name)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def unpack_nbits(data_uint8: np.ndarray, bits: int) -> np.ndarray:
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bit width: {bits}")

    if bits == 8:
        return data_uint8.astype(np.uint8)

    mask = (1 << bits) - 1
    values_per_byte = 8 // bits
    unpacked = [np.bitwise_and(np.right_shift(data_uint8, bits * index), mask) for index in range(values_per_byte)]
    return np.stack(unpacked, axis=-1).reshape(*data_uint8.shape[:-1], -1).astype(np.uint8)


def unpack_zero_points(zero_points: np.ndarray, bits: int, n: int, k_blocks: int) -> np.ndarray:
    unpacked = unpack_nbits(zero_points, bits)
    unpacked = unpacked.reshape(n, -1)
    return unpacked[:, :k_blocks]


def dequantize_qweight(
    quantized_weight: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray | None,
    bits: int,
    block_size: int,
    k: int,
    n: int,
) -> np.ndarray:
    quant_values = unpack_nbits(quantized_weight, bits).reshape(n, -1)
    quant_values = quant_values[:, :k]

    k_blocks = scales.shape[1]
    padded_k = k_blocks * block_size
    if quant_values.shape[1] < padded_k:
        raise ValueError(f"Quantized data is too small for K={k}, block_size={block_size}")

    quant_values = quant_values[:, :padded_k].reshape(n, k_blocks, block_size).astype(np.float32)

    if zero_points is None:
        zero_point_values = np.full((n, k_blocks), 1 << (bits - 1), dtype=np.float32)
    else:
        zero_point_values = unpack_zero_points(zero_points, bits, n, k_blocks).astype(np.float32)

    scale_values = scales.astype(np.float32)[:, :, None]
    dequantized = (quant_values - zero_point_values[:, :, None]) * scale_values
    dequantized = dequantized.reshape(n, padded_k)[:, :k]
    return dequantized.transpose().copy()


def tensor_to_fp8_bytes(array: np.ndarray, torch_dtype: torch.dtype) -> bytes:
    if array.dtype == np.float32:
        source = torch.from_numpy(np.array(array, copy=True))
    else:
        source = torch.from_numpy(np.array(array.astype(np.float32 if array.dtype == np.float64 else np.float16), copy=True))

    fp8_tensor = source.to(torch_dtype)
    return fp8_tensor.view(torch.uint8).cpu().numpy().tobytes()


def make_external_tensor(
    name: str,
    data_type: int,
    dims: tuple[int, ...],
    location: str,
    offset: int,
    length: int,
) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(dims)
    tensor.raw_data = b""
    tensor.data_location = onnx.TensorProto.EXTERNAL
    external_data_helper.set_external_data(
        tensor,
        location=location,
        offset=offset,
        length=length,
    )
    return tensor


def convert_model(
    source_model_path: str = "model_q2/Ternary-Bonsai-8B-ONNX/onnx/model_q2f16.onnx",
    output_model_path: str = "model_q2_to_fp8/model.onnx",
    fp8_format: str = "e4m3fn",
) -> None:
    model_path = Path(source_model_path)
    output_path = Path(output_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Source ONNX model not found: {model_path}. "
            "Run 'uv run python download_q2.py' first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data_path = output_path.with_name(f"{output_path.stem}.onnx_data")
    if output_path.exists():
        output_path.unlink()
    if output_data_path.exists():
        output_data_path.unlink()

    source_root = model_path.parent.parent if model_path.parent.name == "onnx" else model_path.parent
    copy_metadata_files(source_root, output_path.parent)

    fp8_enum, torch_fp8_dtype = FP8_TYPE_MAP[fp8_format]

    print(f"Loading {model_path} into memory...")
    model = onnx.load(str(model_path), load_external_data=True)

    initializer_map = {initializer.name: initializer for initializer in model.graph.initializer}
    removed_initializer_names: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    converted_count = 0
    bytes_written = 0

    print("Dequantizing MatMulNBits weights and rewriting them as dense fp8 MatMul nodes...")
    with output_data_path.open("wb") as output_data_file:
        for node in model.graph.node:
            if node.op_type != "MatMulNBits":
                new_nodes.append(node)
                continue

            attrs = {attribute.name: attribute.i for attribute in node.attribute}
            bits = int(attrs["bits"])
            block_size = int(attrs["block_size"])
            k = int(attrs["K"])
            n = int(attrs["N"])

            if bits != 2:
                raise ValueError(f"This converter only supports q2 inputs, but found bits={bits} on node {node.name!r}.")

            if len(node.input) > 4 and node.input[4]:
                raise ValueError(f"group index input is not supported for node {node.name!r}.")

            quant_initializer = initializer_map[node.input[1]]
            scales_initializer = initializer_map[node.input[2]]
            zero_point_initializer = initializer_map[node.input[3]] if len(node.input) > 3 and node.input[3] else None

            quantized_weight = numpy_helper.to_array(quant_initializer)
            scales = numpy_helper.to_array(scales_initializer)
            zero_points = numpy_helper.to_array(zero_point_initializer) if zero_point_initializer is not None else None

            dense_weight = dequantize_qweight(
                quantized_weight=quantized_weight,
                scales=scales,
                zero_points=zero_points,
                bits=bits,
                block_size=block_size,
                k=k,
                n=n,
            )

            weight_name = f"{sanitize_name(node.name or node.output[0])}__dense_fp8_weight"
            cast_output_name = f"{weight_name}__fp16"
            fp8_bytes = tensor_to_fp8_bytes(dense_weight, torch_fp8_dtype)

            offset = output_data_file.tell()
            output_data_file.write(fp8_bytes)
            new_initializers.append(
                make_external_tensor(
                    name=weight_name,
                    data_type=fp8_enum,
                    dims=(k, n),
                    location=output_data_path.name,
                    offset=offset,
                    length=len(fp8_bytes),
                )
            )

            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[weight_name],
                    outputs=[cast_output_name],
                    name=f"CastBack_{sanitize_name(node.name or node.output[0])}",
                    to=TensorProto.FLOAT16,
                )
            )

            matmul_output_name = node.output[0]
            if len(node.input) > 5 and node.input[5]:
                matmul_output_name = f"{sanitize_name(node.name or node.output[0])}__matmul"

            new_nodes.append(
                helper.make_node(
                    "MatMul",
                    inputs=[node.input[0], cast_output_name],
                    outputs=[matmul_output_name],
                    name=f"Dense_{sanitize_name(node.name or node.output[0])}",
                )
            )

            if len(node.input) > 5 and node.input[5]:
                new_nodes.append(
                    helper.make_node(
                        "Add",
                        inputs=[matmul_output_name, node.input[5]],
                        outputs=list(node.output),
                        name=f"Bias_{sanitize_name(node.name or node.output[0])}",
                    )
                )

            removed_initializer_names.update({node.input[1], node.input[2]})
            if zero_point_initializer is not None:
                removed_initializer_names.add(node.input[3])

            converted_count += 1
            bytes_written += len(fp8_bytes)

            del quantized_weight
            del scales
            del dense_weight
            del fp8_bytes
            if zero_points is not None:
                del zero_points

    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name not in removed_initializer_names
    ]

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(new_initializers)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, 21)
            break
    else:
        model.opset_import.append(helper.make_operatorsetid("", 21))

    print(f"Converted {converted_count} MatMulNBits nodes.")
    print(f"Dense fp8 weight data written: {bytes_written / 1e9:.2f} GB")
    print(f"Saving modified model to {output_path}...")
    onnx.save_model(model, str(output_path))
    print("Conversion completed successfully!")


if __name__ == "__main__":
    arguments = parse_args()
    convert_model(
        source_model_path=arguments.source_model_path,
        output_model_path=arguments.output_model_path,
        fp8_format=arguments.fp8_format,
    )
