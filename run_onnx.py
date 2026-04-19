import argparse
import time

script_start_time = time.perf_counter()

from run_common import add_prompt_generation_args, apply_generation_defaults, build_chat_inputs, build_warmup_inputs, patch_num_logits_to_keep, print_generation_header
from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, TextStreamer


library_load_elapsed = time.perf_counter() - script_start_time


model_id = "onnx-community/Ternary-Bonsai-8B-ONNX"
q4_model_dir = Path("onnx_q2_to_q4")
q8_model_dir = Path("onnx_q2_to_q8")
q2_fp8_model_dir = Path("onnx_q2_to_fp8")
fp8_export_dir = Path("onnx_fp8")
fp32_export_dir = Path("onnx_fp32")
fp16_export_dir = Path("onnx_fp16")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--provider",
    choices=["dml", "cpu"],
    default="dml",
    help="Execution provider to use for ONNX Runtime.",
)
add_prompt_generation_args(parser)
parser.add_argument(
    "--model-source",
    choices=["auto", "fp8", "fp16", "fp32", "q4", "q8", "q2fp8"],
    default="auto",
    help="Which local model artifact to load.",
)
args = parser.parse_args()


def resolve_export_dir(export_dir: Path) -> tuple[Path, str, str] | None:
    if not export_dir.exists():
        return None

    onnx_files = sorted(export_dir.rglob("*.onnx"))
    if not onnx_files:
        return None

    preferred = None
    for file_path in onnx_files:
        if file_path.name in {"model.onnx", "decoder_model_merged.onnx"}:
            preferred = file_path
            break
    if preferred is None:
        preferred = onnx_files[0]

    subfolder = str(preferred.parent.relative_to(export_dir))
    if subfolder == ".":
        subfolder = ""

    return export_dir, subfolder, preferred.name


def resolve_model_artifacts() -> tuple[Path, str, str]:
    fp8_export = resolve_export_dir(fp8_export_dir)
    fp16_export = resolve_export_dir(fp16_export_dir)
    fp32_export = resolve_export_dir(fp32_export_dir)
    q4_model_path = q4_model_dir / "model.onnx"
    q8_model_path = q8_model_dir / "model.onnx"
    q2_fp8_model_path = q2_fp8_model_dir / "model.onnx"

    if args.model_source == "fp8":
        if fp8_export is None:
            raise FileNotFoundError("FP8 ONNX export not found in onnx_fp8.")
        print(f"Using fp8 export from {fp8_export[0]}...")
        return fp8_export

    if args.model_source == "fp16":
        if fp16_export is None:
            raise FileNotFoundError("FP16 ONNX export not found in onnx_fp16.")
        print(f"Using fp16 export from {fp16_export[0]}...")
        return fp16_export

    if args.model_source == "fp32":
        if fp32_export is None:
            raise FileNotFoundError("FP32 ONNX export not found in onnx_fp32.")
        print(f"Using fp32 export from {fp32_export[0]}...")
        return fp32_export

    if args.model_source == "q4":
        if not q4_model_path.exists():
            raise FileNotFoundError(
                "Converted q4 ONNX model not found in onnx_q2_to_q4. "
                "Run 'uv run python convert_q2.py --target-format q4' first."
            )
        print(f"Using converted model from {q4_model_dir}...")
        return q4_model_dir, "", "model.onnx"

    if args.model_source == "q8":
        if not q8_model_path.exists():
            raise FileNotFoundError(
                "Converted q8 ONNX model not found in onnx_q2_to_q8. "
                "Run 'uv run python convert_q2.py --target-format q8' first."
            )
        print(f"Using converted model from {q8_model_dir}...")
        return q8_model_dir, "", "model.onnx"

    if args.model_source == "q2fp8":
        if not q2_fp8_model_path.exists():
            raise FileNotFoundError(
                "Converted q2->fp8 ONNX model not found in onnx_q2_to_fp8. "
                "Run 'uv run python convert_q2_to_fp8.py' first."
            )
        print(f"Using converted fp8 model from {q2_fp8_model_dir}...")
        return q2_fp8_model_dir, "", "model.onnx"

    if args.model_source == "auto" and fp8_export is not None:
        print(f"Using fp8 export from {fp8_export[0]}...")
        return fp8_export

    if args.model_source == "auto" and fp16_export is not None:
        print(f"Using fp16 export from {fp16_export[0]}...")
        return fp16_export

    if args.model_source == "auto" and fp32_export is not None:
        print(f"Using fp32 export from {fp32_export[0]}...")
        return fp32_export

    if q4_model_path.exists():
        print(f"Using converted model from {q4_model_dir}...")
        return q4_model_dir, "", "model.onnx"

    if q2_fp8_model_path.exists():
        print(f"Using converted fp8 model from {q2_fp8_model_dir}...")
        return q2_fp8_model_dir, "", "model.onnx"

    if q8_model_path.exists():
        print(f"Using converted model from {q8_model_dir}...")
        return q8_model_dir, "", "model.onnx"

    raise FileNotFoundError(
        "No usable local model artifacts were found. Create an fp8/fp16/fp32 export, "
        "or run 'uv run python convert_q2_to_fp8.py' / 'uv run python convert_q2.py --target-format q4' / 'uv run python convert_q2.py --target-format q8' to prepare local converted models."
    )


print(f"Loading tokenizer for {model_id}...")
model_load_start_time = time.perf_counter()
model_dir, model_subfolder, model_file_name = resolve_model_artifacts()
tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)

provider = {
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}[args.provider]
print(f"Using provider: {provider}")

print(f"Loading ONNX model {model_file_name}...")
model = ORTModelForCausalLM.from_pretrained(
    model_dir,
    provider=provider,
    use_cache=True,
    subfolder=model_subfolder,
    file_name=model_file_name,
)

apply_generation_defaults(model)
patch_num_logits_to_keep(model)
warmup_inputs, warmup_token_id = build_warmup_inputs(tokenizer)
model.generate(
    **warmup_inputs,
    max_new_tokens=1,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id or warmup_token_id,
    remove_invalid_values=True,
)
model_load_elapsed = time.perf_counter() - model_load_start_time

prompt = args.prompt
print_generation_header(prompt, library_load_elapsed, model_load_elapsed)

inputs = build_chat_inputs(tokenizer, prompt)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_start_time = time.perf_counter()
model.generate(
    **inputs,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    remove_invalid_values=True,
    streamer=streamer
)
generation_elapsed = time.perf_counter() - generation_start_time
total_elapsed = time.perf_counter() - script_start_time

print()
print(f"Generation time: {generation_elapsed:.2f}s")
print(f"Total time: {total_elapsed:.2f}s")
