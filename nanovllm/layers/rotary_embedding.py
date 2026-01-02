from functools import lru_cache
from tinygrad import Tensor


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  x1, x2 = x1.float(), x2.float()
  y1 = x1 * cos - x2 * sin
  y2 = x2 * cos + x1 * sin
  return y1.cat(y2, dim=-1).cast(x.dtype)


class RotaryEmbedding:

  def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float):
    self.head_size = head_size
    assert rotary_dim == head_size
    inv_freq = 1.0 / (base ** (Tensor.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = Tensor.arange(max_position_embeddings).float()
    freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)  # outer product
    cos = freqs.cos()
    sin = freqs.sin()
    cache = cos.cat(sin, dim=-1).unsqueeze(1)
    self.cos_sin_cache = cache

  def __call__(self, positions: Tensor, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return query, key


@lru_cache(1)
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float, rope_scaling: dict | None = None):
  assert rope_scaling is None
  rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
  return rotary_emb
