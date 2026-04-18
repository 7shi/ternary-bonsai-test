import argparse
import time

script_start_time = time.perf_counter()

from inference_common import add_prompt_generation_args, apply_generation_defaults, build_chat_inputs, build_warmup_inputs, patch_num_logits_to_keep, print_generation_header
from transformers import AutoTokenizer, TextStreamer
from optimum.onnxruntime import ORTModelForCausalLM
import torch


library_load_elapsed = time.perf_counter() - script_start_time


parser = argparse.ArgumentParser()
add_prompt_generation_args(parser)
args = parser.parse_args()

print("Loading local q2 packed model...")
print("This direct q2 execution path is expected to fail in the current ONNX Runtime environment.")
local_dir = "model_q2/Ternary-Bonsai-8B-ONNX"

model_load_start = time.perf_counter()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)

provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
print(f"Using provider: {provider}")

model = ORTModelForCausalLM.from_pretrained(
    local_dir,
    provider=provider,
    use_cache=True,
    subfolder="onnx",
    file_name="model_q2f16.onnx"
)
apply_generation_defaults(model)

# Inject the 'num_logits_to_keep' input required by this ONNX model.
patch_num_logits_to_keep(model)
warmup_inputs, warmup_token_id = build_warmup_inputs(tokenizer)
model.generate(
    **warmup_inputs,
    max_new_tokens=1,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id or warmup_token_id,
)
model_load_elapsed = time.perf_counter() - model_load_start

prompt = args.prompt
print_generation_header(prompt, library_load_elapsed, model_load_elapsed)
inputs = build_chat_inputs(tokenizer, prompt)

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
