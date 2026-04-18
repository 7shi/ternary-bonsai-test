import argparse
import time

script_start_time = time.perf_counter()

from inference_common import add_prompt_generation_args, apply_generation_defaults, build_chat_inputs, build_warmup_inputs, print_generation_header
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

library_load_elapsed = time.perf_counter() - script_start_time

DEFAULT_MODEL_ID = "prism-ml/Ternary-Bonsai-8B-unpacked"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-id",
    default=DEFAULT_MODEL_ID,
    help="Hugging Face model ID or local path to the safetensors checkpoint.",
)
add_prompt_generation_args(parser)
parser.add_argument(
    "--dtype",
    choices=["auto", "float32", "float16", "bfloat16"],
    default="auto",
    help="Model dtype to request when loading the safetensors checkpoint.",
)
parser.add_argument(
    "--do-sample",
    action="store_true",
    help="Enable sampling instead of greedy decoding.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature to use when --do-sample is enabled.",
)
args = parser.parse_args()

def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)

model_load_start = time.perf_counter()
print(f"Loading tokenizer for {args.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_id, fix_mistral_regex=True)

print("Loading safetensors model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    dtype=resolve_dtype(args.dtype),
    device_map="cpu",
    low_cpu_mem_usage=True,
)
apply_generation_defaults(model)

model.eval()
warmup_inputs, warmup_token_id = build_warmup_inputs(tokenizer)
with torch.no_grad():
    model.generate(
        **warmup_inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id or warmup_token_id,
    )
model_load_elapsed = time.perf_counter() - model_load_start

inputs = build_chat_inputs(tokenizer, args.prompt)

print_generation_header(args.prompt, library_load_elapsed, model_load_elapsed)

generation_start = time.perf_counter()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
generation_elapsed = time.perf_counter() - generation_start
total_elapsed = time.perf_counter() - script_start_time

prompt_length = inputs["input_ids"].shape[1]
generated_ids = output_ids[0, prompt_length:]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
token_count = int(generated_ids.shape[0])
tokens_per_second = token_count / generation_elapsed if generation_elapsed > 0 else 0.0

print(generated_text)
print()
print(f"Generated tokens: {token_count}")
print(f"Generation time: {generation_elapsed:.2f}s")
print(f"Total time: {total_elapsed:.2f}s")
print(f"Tokens per second: {tokens_per_second:.2f}")
