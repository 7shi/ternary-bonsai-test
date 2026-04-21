import argparse
import re
import time

script_start_time = time.perf_counter()

from run_common import add_prompt_generation_args, apply_generation_defaults, build_chat_inputs, build_warmup_inputs, print_generation_header
from convert_q2_to_fp8 import unpack_nbits
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer

library_load_elapsed = time.perf_counter() - script_start_time


class Q2Linear(nn.Module):
    """Linear layer using block-scaled integer accumulation with Q2 ternary weights.

    Weights are stored as int8 values {-2,-1,0,+1} (after zero_point subtraction).
    Computation: for each block k, partial = x_block @ w_int_block.T, then scale and sum.
    """

    def __init__(self, weight_q: torch.Tensor, scales: torch.Tensor, block_size: int):
        super().__init__()
        self.register_buffer("weight_q", weight_q)  # (N, K) int8
        self.register_buffer("scales", scales)        # (N, n_blocks) float16
        self.block_size = block_size
        self.N, self.K = weight_q.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.K)
        n_blocks = self.K // self.block_size
        x_blocked = x_flat.reshape(-1, n_blocks, self.block_size)             # (batch, n_blocks, block_size)
        w_blocked = self.weight_q.reshape(self.N, n_blocks, self.block_size)  # (N, n_blocks, block_size)
        # Block-wise accumulation: raw[b,n,k] = dot(x[b,k,:], w[n,k,:])
        raw = torch.einsum("bki,nki->bnk",
                           x_blocked.to(torch.float32),
                           w_blocked.to(torch.float32))  # (batch, N, n_blocks)
        out = (raw * self.scales).sum(dim=-1)             # (batch, N)
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.N)


def matmul_node_to_hf_name(node_name: str) -> str | None:
    m = re.match(r"/model/layers\.(\d+)/attn/(\w+)/MatMul_Quant$", node_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.{m.group(2)}.weight"
    m = re.match(r"/model/layers\.(\d+)/mlp/(\w+)/MatMul_Quant$", node_name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.{m.group(2)}.weight"
    if re.match(r"/lm_head/MatMul_Quant$", node_name):
        return "lm_head.weight"
    return None


def norm_init_to_hf_name(init_name: str, num_hidden_layers: int) -> str | None:
    # input_layernorm / post_attention_layernorm: name matches HF directly
    if re.match(r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight$", init_name):
        return init_name
    # q_norm / k_norm: attn.{q,k}_norm.layernorm.weight → self_attn.{q,k}_norm.weight
    m = re.match(r"(model\.layers\.\d+)\.attn\.(q_norm|k_norm)\.layernorm\.weight$", init_name)
    if m:
        return f"{m.group(1)}.self_attn.{m.group(2)}.weight"
    # final norm: model.layers.{num_hidden_layers}.final_norm_layernorm.weight → model.norm.weight
    m = re.match(r"model\.layers\.(\d+)\.final_norm_layernorm\.weight$", init_name)
    if m and int(m.group(1)) == num_hidden_layers:
        return "model.norm.weight"
    return None


def dequantize_block(quant_raw: np.ndarray, scales_raw: np.ndarray, zp_raw: np.ndarray | None,
                     weight_bits: int, zp_bits: int, rows: int, cols: int) -> np.ndarray:
    n_blocks = scales_raw.shape[1]
    block_size = cols // n_blocks
    unpacked = unpack_nbits(quant_raw, weight_bits).reshape(rows, -1)[:, :cols]  # (rows, cols)
    if zp_raw is not None:
        zp = unpack_nbits(zp_raw, zp_bits).reshape(rows, -1)[:, :n_blocks]
    else:
        zp = np.full((rows, n_blocks), 1 << (weight_bits - 1), dtype=np.uint8)
    blocked = unpacked.reshape(rows, n_blocks, block_size).astype(np.float32)
    result = (blocked - zp[:, :, None].astype(np.float32)) * scales_raw[:, :, None].astype(np.float32)
    return result.reshape(rows, cols)


def extract_all_weights(onnx_path: str, num_hidden_layers: int) -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    """Returns (q2_linear_weights, float_weights) extracted from ONNX."""
    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path, load_external_data=True)
    initializer_map = {init.name: init for init in model.graph.initializer}

    q2_weights: dict[str, dict] = {}
    float_weights: dict[str, torch.Tensor] = {}
    unmapped = []

    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            hf_name = matmul_node_to_hf_name(node.name)
            if hf_name is None:
                unmapped.append(node.name)
                continue
            attrs = {attr.name: attr.i for attr in node.attribute}
            bits = int(attrs["bits"])
            block_size = int(attrs["block_size"])
            K = int(attrs["K"])
            N = int(attrs["N"])
            n_blocks = (K + block_size - 1) // block_size
            quant_raw = numpy_helper.to_array(initializer_map[node.input[1]])
            scales_raw = numpy_helper.to_array(initializer_map[node.input[2]])
            zp_init = (initializer_map[node.input[3]]
                       if len(node.input) > 3 and node.input[3] else None)
            unpacked = unpack_nbits(quant_raw, bits).reshape(N, -1)[:, :K]
            if zp_init is not None:
                zp_raw = numpy_helper.to_array(zp_init)
                zp_per_block = unpack_nbits(zp_raw, bits).reshape(N, -1)[:, :n_blocks]
            else:
                zp_per_block = np.full((N, n_blocks), 1 << (bits - 1), dtype=np.uint8)
            zp_expanded = np.repeat(zp_per_block, block_size, axis=1)[:, :K]
            weight_int = (unpacked.astype(np.int16) - zp_expanded.astype(np.int16)).astype(np.int8)
            q2_weights[hf_name] = {
                "weight_q": torch.from_numpy(weight_int),
                "scales": torch.from_numpy(scales_raw.copy()),
                "block_size": block_size,
            }

    # Norm weights (float16)
    for init in model.graph.initializer:
        hf_name = norm_init_to_hf_name(init.name, num_hidden_layers)
        if hf_name is not None:
            arr = numpy_helper.to_array(init)
            float_weights[hf_name] = torch.from_numpy(arr.copy())

    # Embedding: 2-bit weights, 4-bit zero_points
    quant_raw = numpy_helper.to_array(initializer_map["model_embed_tokens_weight_quant"])
    scales_raw = numpy_helper.to_array(initializer_map["model_embed_tokens_weight_scales"])
    zp_raw = numpy_helper.to_array(initializer_map["model_embed_tokens_weight_zp_4b"])
    vocab_size, n_blocks = scales_raw.shape
    hidden_size = n_blocks * 128
    embed = dequantize_block(quant_raw, scales_raw, zp_raw,
                             weight_bits=2, zp_bits=4,
                             rows=vocab_size, cols=hidden_size)
    float_weights["model.embed_tokens.weight"] = torch.from_numpy(embed.astype(np.float16))

    print(f"Extracted {len(q2_weights)} Q2 linear layers, {len(float_weights)} float tensors.")
    if unmapped:
        print(f"Unmapped MatMulNBits nodes ({len(unmapped)}): {unmapped[:5]}{'...' if len(unmapped) > 5 else ''}")
    return q2_weights, float_weights


def get_module_by_path(root: nn.Module, parts: list[str]) -> nn.Module:
    obj = root
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def replace_linears(model: nn.Module, q2_weights: dict[str, dict]) -> None:
    for hf_name, params in q2_weights.items():
        parts = hf_name.split(".")
        parent = get_module_by_path(model, parts[:-2])
        module_name = parts[-2]
        q2_linear = Q2Linear(params["weight_q"], params["scales"], params["block_size"])
        setattr(parent, module_name, q2_linear)
    print(f"Replaced {len(q2_weights)} nn.Linear layers with Q2Linear.")


def print_weight_stats(q2_weights: dict[str, dict]) -> None:
    sample_name = next(iter(q2_weights))
    w = q2_weights[sample_name]["weight_q"]
    vals, counts = w.unique(return_counts=True)
    total = w.numel()
    print(f"\nWeight value distribution ({sample_name}):")
    for v, c in zip(vals.tolist(), counts.tolist()):
        print(f"  {v:+3d}: {c:9d} ({100.0 * c / total:.2f}%)")


parser = argparse.ArgumentParser()
add_prompt_generation_args(parser)
parser.add_argument("--model-dir", default="onnx_q2/Ternary-Bonsai-8B-ONNX",
                    help="Local directory with config.json and tokenizer files.")
parser.add_argument("--onnx-path", default="onnx_q2/Ternary-Bonsai-8B-ONNX/onnx/model_q2f16.onnx")
args = parser.parse_args()

model_load_start = time.perf_counter()

config = AutoConfig.from_pretrained(args.model_dir)
q2_weights, float_weights = extract_all_weights(args.onnx_path, config.num_hidden_layers)
print_weight_stats(q2_weights)

print(f"\nBuilding model from config (no download)...")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
base_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
base_model.load_state_dict(float_weights, strict=False)
apply_generation_defaults(base_model)
replace_linears(base_model, q2_weights)
base_model.eval()

model_load_elapsed = time.perf_counter() - model_load_start

warmup_inputs, warmup_token_id = build_warmup_inputs(tokenizer)
with torch.no_grad():
    base_model.generate(
        **warmup_inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id or warmup_token_id,
    )

print_generation_header(args.prompt, library_load_elapsed, model_load_elapsed)
inputs = build_chat_inputs(tokenizer, args.prompt)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_start = time.perf_counter()
with torch.no_grad():
    output_ids = base_model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
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
