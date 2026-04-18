from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "onnx-community/Ternary-Bonsai-8B-ONNX"
LOCAL_MODEL_DIR = Path("model_q2") / "Ternary-Bonsai-8B-ONNX"
ALLOW_PATTERNS = [
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "onnx/model_q2f16.onnx",
    "onnx/model_q2f16.onnx_data*",
]


def download_model(local_dir: Path = LOCAL_MODEL_DIR) -> Path:
    print(f"Materializing model files into {local_dir}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    return local_dir


if __name__ == "__main__":
    download_model()
