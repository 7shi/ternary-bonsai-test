import argparse
import gc
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper


TARGET_BITS = {
    "q4": 4,
    "q8": 8,
}

DEFAULT_OUTPUTS = {
    "q4": "onnx_q2_to_q4/model.onnx",
    "q8": "onnx_q2_to_q8/model.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-model-path",
        default="onnx_q2/Ternary-Bonsai-8B-ONNX/onnx/model_q2f16.onnx",
        help="Path to the source q2 ONNX model.",
    )
    parser.add_argument(
        "--target-format",
        choices=sorted(TARGET_BITS),
        default="q8",
        help="Target MatMulNBits format to generate.",
    )
    parser.add_argument(
        "--output-model-path",
        help="Path to write the converted ONNX model. Defaults to onnx_q2_to_q4/model.onnx or onnx_q2_to_q8/model.onnx.",
    )
    return parser.parse_args()


def unpack_nbit(data_uint8: np.ndarray, bits: int) -> np.ndarray:
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bit width: {bits}")

    if bits == 8:
        return data_uint8.astype(np.uint8)

    mask = (1 << bits) - 1
    values_per_byte = 8 // bits
    unpacked = [np.bitwise_and(np.right_shift(data_uint8, bits * index), mask) for index in range(values_per_byte)]
    return np.stack(unpacked, axis=-1).reshape(*data_uint8.shape[:-1], -1).astype(np.uint8)


def pack_nbit(data_uint8: np.ndarray, bits: int) -> np.ndarray:
    if bits not in (4, 8):
        raise ValueError(f"Unsupported target bit width: {bits}")

    if bits == 8:
        return data_uint8.astype(np.uint8)

    values_per_byte = 8 // bits
    flat = data_uint8.reshape(-1)
    padding = (-flat.size) % values_per_byte
    if padding:
        flat = np.pad(flat, (0, padding), constant_values=0)

    packed = np.zeros(flat.size // values_per_byte, dtype=np.uint8)
    mask = (1 << bits) - 1
    for index in range(values_per_byte):
        packed |= np.left_shift(np.bitwise_and(flat[index::values_per_byte], mask), bits * index)
    return packed.reshape(*data_uint8.shape[:-1], -1)


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


def convert_model(source_model_path: str, output_model_path: str, target_bits: int) -> None:
    model_path = Path(source_model_path)
    output_path = Path(output_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Source ONNX model not found: {model_path}. "
            "Run 'uv run python download_q2.py' first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data_path = output_path.with_name(f"{output_path.stem}.onnx_data")

    source_root = model_path.parent.parent if model_path.parent.name == "onnx" else model_path.parent
    copy_metadata_files(source_root, output_path.parent)

    print(f"Loading {model_path} into memory...")
    model = onnx.load(str(model_path), load_external_data=True)

    print("Finding target nodes and initializers...")
    target_b_names = set()
    target_zp_names = set()

    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            for attr in node.attribute:
                if attr.name == "bits":
                    attr.i = target_bits

            target_b_names.add(node.input[1])
            if len(node.input) > 3 and node.input[3]:
                target_zp_names.add(node.input[3])

    print(f"Repacking {len(target_b_names)} MatMulNBits parameters from 2-bit to {target_bits}-bit...")
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name not in target_b_names and initializer.name not in target_zp_names:
            continue

        data = numpy_helper.to_array(initializer)
        unpacked_data = unpack_nbit(data, bits=2)
        repacked_data = pack_nbit(unpacked_data, bits=target_bits)

        new_initializer = numpy_helper.from_array(repacked_data, name=initializer.name)
        model.graph.initializer[index].CopyFrom(new_initializer)

        del data
        del unpacked_data
        del repacked_data

    gc.collect()

    estimated_size = "~7GB" if target_bits == 8 else "~3.5GB"
    print(f"Saving modified model to {output_path} (This will take a while and produce a {estimated_size} file)...")
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


def main() -> None:
    args = parse_args()
    target_bits = TARGET_BITS[args.target_format]
    output_model_path = args.output_model_path or DEFAULT_OUTPUTS[args.target_format]
    convert_model(args.source_model_path, output_model_path, target_bits)


if __name__ == "__main__":
    main()