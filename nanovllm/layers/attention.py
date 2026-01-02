from tinygrad import Tensor
from nanovllm.utils.context import get_context


class Attention:

  def __init__(self, num_heads, head_dim, scale, num_kv_heads):
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.scale = scale
    self.num_kv_heads = num_kv_heads
    # Simple per-layer cache instead of paged cache
    self.cached_k = None
    self.cached_v = None
    # For compatibility with model_runner
    self.k_cache = None
    self.v_cache = None

  def reset_cache(self):
    """Clear the KV cache"""
    self.cached_k = None
    self.cached_v = None

  def __call__(self, q: Tensor, k: Tensor, v: Tensor):
    context = get_context()

    # q: (N, num_heads, head_dim)
    # k: (N, num_kv_heads, head_dim)
    # v: (N, num_kv_heads, head_dim)

    # In decode mode, use cached K and V
    if not context.is_prefill:
      # Decode: concatenate cached K,V with new K,V
      if self.cached_k is not None and self.cached_v is not None:
        k = Tensor.cat(self.cached_k, k, dim=0)
        v = Tensor.cat(self.cached_v, v, dim=0)

      # Update cache with the concatenated K,V
      self.cached_k = k
      self.cached_v = v
    else:
      # Prefill: just cache the K,V for future use
      self.cached_k = k
      self.cached_v = v

    N = k.shape[0]

    # Handle grouped query attention: repeat k,v if num_heads > num_kv_heads
    if self.num_heads != self.num_kv_heads:
      n_rep = self.num_heads // self.num_kv_heads
      k = k.repeat_interleave(n_rep, dim=1)
      v = v.repeat_interleave(n_rep, dim=1)

    # Reshape to (num_heads, N, head_dim) for batched matmul
    q = q.transpose(0, 1)  # (num_heads, N, head_dim)
    k = k.transpose(0, 1)  # (num_heads, N, head_dim)
    v = v.transpose(0, 1)  # (num_heads, N, head_dim)

    # Compute attention scores: (num_heads, q_len, kv_len)
    attn_scores = (q @ k.transpose(-2, -1)) * self.scale

    # Apply causal mask
    q_len = q.shape[1]
    kv_len = k.shape[1]

    # For decode, q_len=1 and we attend to all kv_len positions
    # For prefill, we need standard causal masking
    if q_len == 1:
      # Decode: attend to all positions (no masking needed)
      pass
    else:
      # Prefill: standard causal masking
      mask = Tensor.ones(q_len, kv_len).tril(0).reshape(1, q_len, kv_len)
      attn_scores = mask.where(attn_scores, float('-inf'))

    # Softmax and weighted sum
    attn_probs = attn_scores.softmax(axis=-1)
    o = attn_probs @ v  # (num_heads, q_len, head_dim)

    # Transpose back to (q_len, num_heads, head_dim)
    o = o.transpose(0, 1)

    return o
