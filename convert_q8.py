import onnx
from onnx import numpy_helper
import numpy as np
import gc
from pathlib import Path
import shutil

def unpack_2bit_to_8bit(data_uint8):
    # 下位ビットから順に要素が格納されていると仮定
    val0 = np.bitwise_and(data_uint8, 0x03)
    val1 = np.bitwise_and(np.right_shift(data_uint8, 2), 0x03)
    val2 = np.bitwise_and(np.right_shift(data_uint8, 4), 0x03)
    val3 = np.bitwise_and(np.right_shift(data_uint8, 6), 0x03)
    
    # 最後の次元の末尾で結合してフラット化 [..., 32] -> [..., 32, 4] -> [..., 128]
    unpacked = np.stack([val0, val1, val2, val3], axis=-1)
    unpacked = unpacked.reshape(*data_uint8.shape[:-1], -1)
    return unpacked.astype(np.uint8)


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


def convert_model(source_model_path="model_cache/Ternary-Bonsai-8B-ONNX/onnx/model_q2f16.onnx", output_model_path="model_q2_to_q8/model.onnx"):
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
    # BとZPのinitializerの名前を収集
    target_b_names = set()
    target_zp_names = set()
    
    for node in model.graph.node:
        if node.op_type == 'MatMulNBits':
            # bits 属性を 8 に変更
            for attr in node.attribute:
                if attr.name == 'bits':
                    attr.i = 8
            
            target_b_names.add(node.input[1])
            if len(node.input) > 3:
                target_zp_names.add(node.input[3])

    print(f"Unpacking {len(target_b_names)} MatMulNBits parameters from 2-bit to 8-bit...")
    # initializer を直接書き換える
    for i, init in enumerate(model.graph.initializer):
        if init.name in target_b_names or init.name in target_zp_names:
            data = numpy_helper.to_array(init)
            
            # アンパック処理
            unpacked_data = unpack_2bit_to_8bit(data)
            
            # 新しいテンソルに書き換え
            new_init = numpy_helper.from_array(unpacked_data, name=init.name)
            model.graph.initializer[i].CopyFrom(new_init)
            
            # メモリ解放
            del data
            del unpacked_data
    
    gc.collect()
            
    print(f"Saving modified model to {output_path} (This will take a while and produce a ~7GB file)...")
    if output_path.exists():
        output_path.unlink()
    if output_data_path.exists():
        output_data_path.unlink()

    # 外部データとして保存（メモリ節約のため）
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_data_path.name,
    )
    print("Conversion completed successfully!")

if __name__ == "__main__":
    convert_model()
