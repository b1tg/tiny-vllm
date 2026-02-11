# Changelog

## 2025-02-11: Performance optimization (~12x speedup)

Rewrote tiny-vllm to follow tinygrad's LLM inference patterns, achieving ~107 tok/s decode on AMD GPU (up from ~0.3 tok/s on CPU).

### Changes

- **models/qwen3.py**: Rewrote model from vLLM-style (flat tensors + context manager) to tinygrad-style (B,T,D tensors + `start_pos`). KV cache uses inline `assign().realize()` for in-place updates. Uses `scaled_dot_product_attention(enable_gqa=True)`.
- **engine/model_runner.py**: Added `UOp.variable` + `.bind()` for symbolic start_pos (avoids kernel recompilation per decode step). Added `TinyJit` for decode forward pass. Removed `Device.DEFAULT = "CPU"` hardcode.
- **layers/rotary_embedding.py**: Replaced custom RotaryEmbedding class with cached `precompute_freqs_cis` function.
- **layers/layernorm.py**: Simplified RMSNorm, removed fused add+norm.
- **layers/linear.py**: Added dtype casting in weight loaders (bfloat16 -> float16).
- **utils/loader.py**: Added HuggingFace weight name mapping (`_map_path`), bfloat16 -> float16 conversion.

### Limitations

- Single sequence only (`max_num_seqs=1`), no batching.
- First prefill per unique prompt length requires kernel compilation (~2-3s).
