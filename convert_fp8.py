import argparse
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto
import torch


FLOAT_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
}

FP8_TYPE_MAP = {
    "e4m3fn": (TensorProto.FLOAT8E4M3FN, torch.float8_e4m3fn),
    "e5m2": (TensorProto.FLOAT8E5M2, torch.float8_e5m2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-model-path",
        default="onnx_fp16/model.onnx",
        help="Path to the source ONNX model.",
    )
    parser.add_argument(
        "--output-model-path",
        default="onnx_fp8/model.onnx",
        help="Path to write the converted ONNX model.",
    )
    parser.add_argument(
        "--fp8-format",
        choices=sorted(FP8_TYPE_MAP),
        default="e4m3fn",
        help="FP8 storage format to use for converted initializers.",
    )
    parser.add_argument(
        "--min-elements",
        type=int,
        default=1024,
        help="Only convert initializers with at least this many elements.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def tensor_to_fp8_initializer(
    initializer: onnx.TensorProto,
    array: np.ndarray,
    fp8_enum: int,
    torch_dtype: torch.dtype,
) -> onnx.TensorProto:
    if array.dtype == np.float32:
        torch_source = torch.from_numpy(np.array(array, copy=True))
    else:
        torch_source = torch.from_numpy(
            np.array(array.astype(np.float32 if array.dtype == np.float64 else np.float16), copy=True)
        )

    fp8_tensor = torch_source.to(torch_dtype)

    new_initializer = onnx.TensorProto()
    new_initializer.name = initializer.name
    new_initializer.data_type = fp8_enum
    new_initializer.dims.extend(initializer.dims)
    new_initializer.raw_data = fp8_tensor.view(torch.uint8).cpu().numpy().tobytes()
    return new_initializer


def convert_model(
    source_model_path: str = "onnx_fp16/model.onnx",
    output_model_path: str = "onnx_fp8/model.onnx",
    fp8_format: str = "e4m3fn",
    min_elements: int = 1024,
) -> None:
    model_path = Path(source_model_path)
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data_path = output_path.with_name(f"{output_path.stem}.onnx_data")

    fp8_enum, torch_fp8_dtype = FP8_TYPE_MAP[fp8_format]

    for metadata_name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
    ]:
        source_metadata = model_path.parent / metadata_name
        if source_metadata.exists():
            shutil.copy2(source_metadata, output_path.parent / metadata_name)

    print(f"Loading {model_path} into memory...")
    model = onnx.load(str(model_path), load_external_data=True)

    converted_names = []
    cast_nodes = []

    print("Converting eligible initializers to fp8 storage...")
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type not in FLOAT_TYPES:
            continue

        array = onnx.numpy_helper.to_array(initializer)
        if array.size < min_elements:
            continue

        cast_output = f"{initializer.name}__cast_back"
        original_type = initializer.data_type

        new_initializer = tensor_to_fp8_initializer(
            initializer,
            array,
            fp8_enum,
            torch_fp8_dtype,
        )
        model.graph.initializer[index].CopyFrom(new_initializer)

        for node in model.graph.node:
            for input_index, input_name in enumerate(node.input):
                if input_name == initializer.name:
                    node.input[input_index] = cast_output

        cast_nodes.append(
            onnx.helper.make_node(
                "Cast",
                inputs=[initializer.name],
                outputs=[cast_output],
                name=f"CastBack_{sanitize_name(initializer.name)}",
                to=original_type,
            )
        )
        converted_names.append(initializer.name)

    if not converted_names:
        raise RuntimeError("No eligible floating-point initializers were converted to fp8.")

    existing_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(cast_nodes)
    model.graph.node.extend(existing_nodes)

    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, 21)
            break
    else:
        model.opset_import.append(onnx.helper.make_operatorsetid("", 21))

    print(f"Converted {len(converted_names)} initializers.")
    print(f"Saving modified model to {output_path}...")
    if output_path.exists():
        output_path.unlink()
    if output_data_path.exists():
        output_data_path.unlink()
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_data_path.name,
    )
    print("Conversion completed successfully!")


if __name__ == "__main__":
    arguments = parse_args()
    convert_model(
        source_model_path=arguments.source_model_path,
        output_model_path=arguments.output_model_path,
        fp8_format=arguments.fp8_format,
        min_elements=arguments.min_elements,
    )
