# ONNX Q2 Model: Implementation Reference

This document records the structural details needed to extract weights from
`model_q2f16.onnx` and run them in PyTorch without ONNX Runtime.

---

## Graph Overview

The model is Qwen3 (36 transformer layers, hidden_size=4096) exported with
Microsoft custom ops. Node type counts:

| Op type | Count | Role |
|---|---|---|
| `MatMulNBits` | 253 | 2-bit quantized linear projections |
| `Reshape` | 146 | Tensor shape adjustments |
| `SimplifiedLayerNormalization` | 73 | RMSNorm (plain: q_norm, k_norm, final norm) |
| `SkipSimplifiedLayerNormalization` | 72 | RMSNorm with residual add (input/post-attn norms) |
| `GroupQueryAttention` | 36 | Fused GQA + KV cache + RoPE |
| `GatherBlockQuantized` | 1 | 2-bit quantized embedding lookup |

---

## MatMulNBits: Linear Layer Weights

### Node attributes

| Attribute | Meaning |
|---|---|
| `bits` | Quantization bit width (always `2` in this model) |
| `block_size` | Scale granularity (always `128`) |
| `K` | Input dimension |
| `N` | Output dimension |

### Node inputs

| Index | Content |
|---|---|
| `[1]` | Packed quantized weights (uint8, 2-bit packed) |
| `[2]` | Per-block scales (float16, shape `(N, K // block_size)`) |
| `[3]` | Per-block zero-points (uint8, 2-bit packed, shape `(N, K // block_size // 4)`) |

### Weight unpacking

Each uint8 byte stores 4 weight values (2 bits each), packed from LSB:

```
byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
```

To recover integer weight values:

```python
unpacked = unpack_nbits(quant_raw, bits=2)   # uint8, values 0–3
unpacked = unpacked.reshape(N, -1)[:, :K]    # (N, K)
zp       = unpack_nbits(zp_raw, bits=2).reshape(N, -1)[:, :n_blocks]  # (N, n_blocks)
zp_full  = np.repeat(zp, block_size, axis=1)[:, :K]
weight_int = (unpacked.astype(np.int16) - zp_full.astype(np.int16)).astype(np.int8)
```

Default zero-point when `input[3]` is absent: `1 << (bits - 1)` = **2**.

### Ternary value distribution

After zero-point subtraction the model uses exactly three values:

```
-1 : ~32.7%
 0 : ~34.5%
+1 : ~32.7%
```

Value `-2` (or any other) is never present, confirming true ternary quantization.

### Block-scaled computation

The forward pass mirrors ONNX `MatMulNBits` semantics:

```
output = sum_k( (x_block_k @ w_int_block_k.T) * scale_k )
```

In PyTorch:

```python
x_blocked = x.reshape(batch, n_blocks, block_size)          # (batch, n_blocks, block_size)
w_blocked = weight_int.reshape(N, n_blocks, block_size)     # (N, n_blocks, block_size)
raw = torch.einsum("bki,nki->bnk", x_blocked.float(), w_blocked.float())  # (batch, N, n_blocks)
out = (raw * scales).sum(dim=-1)                            # (batch, N)
```

`scales` shape is `(N, n_blocks)`; broadcasting over the batch dimension is automatic.

### Node name → HuggingFace weight name

| ONNX node name pattern | HuggingFace parameter name |
|---|---|
| `/model/layers.{L}/attn/{proj}/MatMul_Quant` | `model.layers.{L}.self_attn.{proj}.weight` |
| `/model/layers.{L}/mlp/{proj}/MatMul_Quant` | `model.layers.{L}.mlp.{proj}.weight` |
| `/lm_head/MatMul_Quant` | `lm_head.weight` |

Note: the ONNX uses `attn` where HuggingFace uses `self_attn`.

Weight shape from `dequantize_qweight` is `(K, N)` (transposed). PyTorch
`nn.Linear.weight` is `(N, K)`, so transpose with `.T` before loading.
For the integer `Q2Linear` implementation the weights are stored as-is in
`(N, K)` form; no transpose is needed because the einsum handles orientation.

---

## Norm Weights (float16 initializers)

Norm initializer names use **dots** (not underscores), directly matching the
PyTorch module hierarchy at export time — with two exceptions:

| ONNX initializer name | HuggingFace parameter name |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | same |
| `model.layers.{L}.post_attention_layernorm.weight` | same |
| `model.layers.{L}.attn.q_norm.layernorm.weight` | `model.layers.{L}.self_attn.q_norm.weight` |
| `model.layers.{L}.attn.k_norm.layernorm.weight` | `model.layers.{L}.self_attn.k_norm.weight` |
| `model.layers.{num_hidden_layers}.final_norm_layernorm.weight` | `model.norm.weight` |

The final norm is stored under a pseudo-layer index equal to `num_hidden_layers`
(36 for this model), not under `model.norm` directly.

`SimplifiedLayerNormalization` and `SkipSimplifiedLayerNormalization` differ by
whether a residual skip connection is fused in:

- **`SimplifiedLayerNormalization`**: q_norm, k_norm, and the final norm
  (no residual). Weight is `node.input[1]`.
- **`SkipSimplifiedLayerNormalization`**: input_layernorm and
  post_attention_layernorm (residual fused). Weight is `node.input[2]`.

---

## Embedding: GatherBlockQuantized

The token embedding uses the same 2-bit block quantization as the linear
layers, but zero-points are stored as **4-bit** packed values (suffix `_4b`).

### Initializer names and shapes

| Initializer | Shape | Content |
|---|---|---|
| `model_embed_tokens_weight_quant` | `(151669, 1024)` | 2-bit packed weights |
| `model_embed_tokens_weight_scales` | `(151669, 32)` | float16 scales |
| `model_embed_tokens_weight_zp_4b` | `(151669, 16)` | 4-bit packed zero-points |

Dimensions: vocab_size=151669, hidden_size=4096, block_size=128, n_blocks=32.
Packing: 4096 values × 2 bits = 1024 bytes; 32 ZPs × 4 bits = 16 bytes.

### Dequantization

```python
unpacked = unpack_nbits(quant_raw, 2).reshape(vocab_size, -1)[:, :hidden_size]
zp       = unpack_nbits(zp_raw,   4).reshape(vocab_size, -1)[:, :n_blocks]
blocked  = unpacked.reshape(vocab_size, n_blocks, block_size).astype(np.float32)
embed    = (blocked - zp[:, :, None]) * scales[:, :, None]   # float32
embed    = embed.reshape(vocab_size, hidden_size).astype(np.float16)
```

---

## Loading Without HuggingFace Download

All weights needed for inference are present in the ONNX file. The model
architecture can be instantiated locally from `config.json`:

```python
config     = AutoConfig.from_pretrained("onnx_q2/Ternary-Bonsai-8B-ONNX")
tokenizer  = AutoTokenizer.from_pretrained("onnx_q2/Ternary-Bonsai-8B-ONNX")
model      = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
```

Load norm and embedding weights via `load_state_dict(..., strict=False)`, then
replace each `nn.Linear` projection with a custom `Q2Linear` module that holds
`(N, K)` int8 weights and `(N, n_blocks)` float16 scales.

---

## CPU Inference Performance: Optimization Experiments

Environment: WSL2, 32 GB RAM, CPU only (no GPU).

### Baseline

Original implementation using `torch.einsum("bki,nki->bnk", ...)` with int8 weights
converted to float32 per forward call:

```
Generation time: ~91s for 10 tokens → 0.11 tok/s
```

### Approaches tried and results

| Approach | tok/s | Notes |
|---|---|---|
| `einsum` float32 (baseline) | 0.11 | int8 → float32 per call |
| pos/neg float16 masks + `bmm` | OOM | 4× int8 memory (2 float16 tensors) |
| float16 `bmm` (precomputed layout) | 0.02 | No AVX-512 FP16 → slow fallback |
| int8 layout + float32 `bmm` | 0.11 | Same as baseline |
| + `torch.compile` | 0.11 | No effect |
| `torch._int_mm` loop (32 blocks) | 0.10 | Python loop overhead dominates |
| Pre-scaled float16 + single `F.linear` | 0.11 | Loop eliminated, still same |

### Why 0.11 tok/s is the ceiling

For a single token (batch=1), measured time per `Q2Linear` call averages ~35 ms,
while memory-bandwidth theory predicts ~0.4–3 ms depending on layer size.
The **87×** gap is PyTorch CPU dispatch overhead: thread management, tensor
allocation/deallocation, and kernel launch cost per operation.

This overhead is not addressable within pure Python/PyTorch regardless of the
linear-algebra formulation used.

### Key findings

- **float16 `bmm` on CPU without AVX-512 FP16** falls back to a slow emulation
  path and is 5× slower than float32.
- **`torch.compile`** does not help for CPU-dispatch-overhead-bound workloads.
- **`torch._int_mm` in a Python loop** (32 blocks × 253 layers = 8096 calls/token)
  adds loop overhead that negates the int8 GEMM benefit.
- **Pre-scaling** (absorbing block scales into float16 weights for a single SGEMV)
  eliminates all loops but still hits the same dispatch ceiling.
- The **current implementation** uses pre-scaled float16 + `torch.compile` +
  single `F.linear` per layer.

### Path to faster CPU inference

To exceed ~0.5 tok/s on CPU, the dispatch overhead must be bypassed:

- **llama.cpp / GGUF**: purpose-built C++ CPU inference, 50–100× faster.
- **Custom C extension**: eliminates PyTorch kernel launch cost.
- **Batched inference**: amortise overhead over multiple parallel prompts.
