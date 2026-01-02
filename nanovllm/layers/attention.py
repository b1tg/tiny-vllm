from tinygrad import Tensor
from nanovllm.utils.context import get_context


class Attention:

  def __init__(self, num_heads, head_dim, scale, num_kv_heads):
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.scale = scale
    self.num_kv_heads = num_kv_heads
    self.k_cache = None
    self.v_cache = None

  def __call__(self, q: Tensor, k: Tensor, v: Tensor):
    context = get_context()

    # For now, just use simple attention without complex KV cache logic
    # TODO: Implement paged attention properly

    # q: (N, num_heads, head_dim)
    # k: (N, num_kv_heads, head_dim)
    # v: (N, num_kv_heads, head_dim)

    # Handle grouped query attention: repeat k,v if num_heads > num_kv_heads
    if self.num_heads != self.num_kv_heads:
      n_rep = self.num_heads // self.num_kv_heads
      k = k.repeat_interleave(n_rep, dim=1)
      v = v.repeat_interleave(n_rep, dim=1)

    # Reshape to (num_heads, N, head_dim) for batched matmul
    q = q.transpose(0, 1)  # (num_heads, N, head_dim)
    k = k.transpose(0, 1)  # (num_heads, N, head_dim)
    v = v.transpose(0, 1)  # (num_heads, N, head_dim)

    # Compute attention scores: (num_heads, N, N)
    attn_scores = (q @ k.transpose(-2, -1)) * self.scale

    # Apply causal mask
    N = q.shape[1]
    mask = Tensor.ones(N, N).tril(0).reshape(1, N, N)
    attn_scores = mask.where(attn_scores, float('-inf'))

    # Softmax and weighted sum
    attn_probs = attn_scores.softmax(axis=-1)
    o = attn_probs @ v  # (num_heads, N, head_dim)

    # Transpose back to (N, num_heads, head_dim)
    o = o.transpose(0, 1)

    return o
