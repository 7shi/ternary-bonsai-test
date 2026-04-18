import argparse
import time

script_start_time = time.perf_counter()

from transformers import AutoTokenizer, TextStreamer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

library_load_elapsed = time.perf_counter() - script_start_time

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="AIの未来について考えてください。")
parser.add_argument("--max-new-tokens", type=int, default=10)
args = parser.parse_args()

print("Loading local converted 8-bit model...")
local_dir = "model_q2_to_q8"

model_load_start = time.perf_counter()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)

provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
print(f"Using provider: {provider}")

model = ORTModelForCausalLM.from_pretrained(
    local_dir,
    provider=provider,
    use_cache=True,
    file_name="model.onnx"
)

# ONNXモデルが要求する 'num_logits_to_keep' 入力を内部的に注入するパッチ
original_prepare = model._prepare_onnx_inputs

def patched_prepare(use_torch, model_inputs):
    if "num_logits_to_keep" not in model_inputs:
        model_inputs["num_logits_to_keep"] = torch.tensor(1, dtype=torch.int64)
    return original_prepare(use_torch, model_inputs)

model._prepare_onnx_inputs = patched_prepare
model_load_elapsed = time.perf_counter() - model_load_start

prompt = args.prompt
print(">", prompt)
print()
inputs = tokenizer(prompt, return_tensors="pt")

print(f"Library load time: {library_load_elapsed:.2f}s")
print(f"Model load time: {model_load_elapsed:.2f}s")
print("Generating...")

# ストリーマーの初期化
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_start = time.perf_counter()
output_ids = model.generate(
    **inputs,
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer
)
generation_elapsed = time.perf_counter() - generation_start
total_elapsed = time.perf_counter() - script_start_time

prompt_length = inputs["input_ids"].shape[1]
generated_ids = output_ids[0, prompt_length:]
token_count = int(generated_ids.shape[0])
tokens_per_second = token_count / generation_elapsed if generation_elapsed > 0 else 0.0

print()
print(f"Generated tokens: {token_count}")
print(f"Generation time: {generation_elapsed:.2f}s")
print(f"Total time: {total_elapsed:.2f}s")
print(f"Tokens per second: {tokens_per_second:.2f}")
