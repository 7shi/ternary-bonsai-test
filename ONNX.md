# Investigation: Why DirectML Output Quality Differs Between onnx_fp8 and onnx_q2_to_fp8

## Summary

`onnx_q2_to_fp8` and `onnx_fp8` store weights as dense fp8 tensors via the same `Cast` + `MatMul` pattern, so they should theoretically produce identical results. However, `onnx_q2_to_fp8` produces garbled English token sequences under DirectML while `onnx_fp8` does not. The root cause is not the weight representation but the **ONNX graph structure**: `onnx_q2_to_fp8` inherits the q2 source graph intact, which relies on `com.microsoft` custom ops that DirectML does not fully support.

---

## Graph Structure Comparison

### onnx_fp8

- Opset: standard ONNX domain only (`""`, version 21)
- All ops are standard: `MatMul`, `Cos`, `Sin`, `Softmax`, `ReduceMean`, `Add`, `Mul`, etc.
- No custom op domains

### onnx_q2_to_fp8

- Opsets: standard ONNX (`""`, version 21) **plus** `com.microsoft` (version 1)
- `convert_q2_to_fp8.py` only replaces `MatMulNBits` nodes with dense fp8 `MatMul` nodes
- All other nodes from the q2 source graph remain untouched, including the following `com.microsoft` custom ops:

| Op | Count | Role |
|----|------:|------|
| `GroupQueryAttention` | 36 | Fused attention + KV cache + RoPE |
| `SkipSimplifiedLayerNormalization` | 72 | Add(residual) + RMSNorm, outputs skip-connection tensor |
| `SimplifiedLayerNormalization` | 1 | Plain RMSNorm |
| `GatherBlockQuantized` | 1 | Embedding lookup with 4-bit block-quantized weights |

The same graph structure is also inherited by `onnx_q2_to_q4` and `onnx_q2_to_q8`, which explains why all three q2-derived models exhibit garbled output under DirectML.

---

## Root Cause

`GroupQueryAttention` is the primary culprit. It is a fused kernel that encapsulates:

- **Grouped Query Attention** (`num_heads=32`, `kv_num_heads=8`, so K/V are repeated ×4)
- **KV cache management** using `seqlens_k` and `total_sequence_length` dynamic inputs
- **RoPE** applied internally via `cos_cache`/`sin_cache` (`do_rotary=1`, non-interleaved)
- **Attention bias** addition (11th input)

CPUExecutionProvider ships a reference implementation for these `com.microsoft` ops inside ONNX Runtime. DirectML, however, either lacks a native kernel for `GroupQueryAttention` or has an incomplete implementation. When DirectML encounters this op it falls back to an incorrect path, producing numerically wrong attention outputs, which cascade into garbled token sequences from the first generated token onward.

`GatherBlockQuantized` (4-bit block-quantized embedding) is a secondary concern: its DirectML handling may also be incorrect, though the attention failure alone is sufficient to break generation.

The same mechanism affects `onnx_q2_to_q8` and `onnx_q2_to_q4` since neither conversion script touches the attention or normalization nodes.

---

## Why CPU Is Unaffected

CPUExecutionProvider has well-tested reference kernels for `GroupQueryAttention` and the other `com.microsoft` ops. That is why `onnx_q2_to_fp8 / CPU`, `onnx_q2_to_q8 / CPU`, and `onnx_q2_to_q4 / CPU` all produce correct Japanese output.

---

## Feasibility of Converting the Graph to Standard Ops

Converting `onnx_q2_to_fp8` to use only standard ONNX ops (to match `onnx_fp8`) is technically possible but has varying difficulty per op:

| Op | Difficulty | Notes |
|----|:----------:|-------|
| `SimplifiedLayerNormalization` | Low | Direct RMSNorm expansion: `Pow → ReduceMean → Add(ε) → Sqrt → Div → Mul` |
| `SkipSimplifiedLayerNormalization` | Low–Medium | `Add(input, skip)` then RMSNorm; the intermediate sum must be wired to two consumers |
| `GatherBlockQuantized` | Low | Dequantize 4-bit weights offline → dense `Gather` |
| `GroupQueryAttention` | High | Must manually implement GQA head expansion, dynamic KV cache concat, RoPE rotation, masked scaled dot-product attention, and bias addition — 36 times across all layers |

The high cost of expanding `GroupQueryAttention` makes this approach impractical compared to simply using `onnx_fp8` (generated from safetensors via `optimum-cli export onnx`) when DirectML + fp8 quality is required.

---

## Conclusions

1. **Weight format is not the differentiator.** Both models store weights as dense fp8, but the graphs are entirely different.
2. **The q2 source graph uses `com.microsoft` custom ops** that CPUExecutionProvider supports but DirectML does not implement correctly.
3. **All q2-derived models** (`onnx_q2_to_fp8`, `onnx_q2_to_q4`, `onnx_q2_to_q8`) share this limitation because none of the conversion scripts rewrite the attention or normalization nodes.
4. **For DirectML with reliable quality**, use `onnx_fp16` or `onnx_fp8` (safetensors-derived, standard ONNX ops only).
5. **For CPU with minimum model size**, `onnx_q2_to_q4 / CPU` remains the best trade-off: fast load, fast generation, correct output, and benefits from OS file cache on repeated runs.
