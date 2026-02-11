from tinygrad import Tensor, UOp, nn
from transformers import Qwen3Config

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import precompute_freqs_cis, apply_rope
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear


class Qwen3Block:

  def __init__(self, config: Qwen3Config):
    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, 'head_dim', None) or dim // n_heads
    norm_eps = config.rms_norm_eps
    rope_theta = getattr(config, "rope_theta", 1000000)
    max_context = config.max_position_embeddings
    qkv_bias = getattr(config, 'attention_bias', True)

    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.rope_theta = rope_theta
    self.max_context = max_context
    self.qk_norm = 0 if qkv_bias else head_dim

    # Attention projections
    self.attn_qkv = QKVParallelLinear(dim, head_dim, n_heads, n_kv_heads, bias=qkv_bias)
    self.attn_output = RowParallelLinear(n_heads * head_dim, dim, bias=False)

    # RMSNorms
    self.attn_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)
    if self.qk_norm:
      self.attn_q_norm = RMSNorm(head_dim, norm_eps)
      self.attn_k_norm = RMSNorm(head_dim, norm_eps)

    # Feed-forward
    self.ffn_gate_up = MergedColumnParallelLinear(dim, [hidden_dim] * 2, bias=False)
    self.ffn_down = RowParallelLinear(hidden_dim, dim, bias=False)

  def _attention(self, x: Tensor, start_pos: int | UOp) -> Tensor:
    x_norm = self.attn_norm(x)
    B, T, _ = x.shape

    qkv = self.attn_qkv(x_norm)
    q_size = self.n_heads * self.head_dim
    kv_size = self.n_kv_heads * self.head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

    if self.qk_norm:
      q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs_cis = precompute_freqs_cis(self.head_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    # KV cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    attn = self.attn_output(attn)
    return x + attn

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    gate_up = self.ffn_gate_up(h_norm)
    gate, up = gate_up.chunk(2, -1)
    return h + self.ffn_down(gate.silu().contiguous() * up)

  def __call__(self, x: Tensor, start_pos: int | UOp):
    return self._feed_forward(self._attention(x, start_pos)).contiguous()


class Qwen3ForCausalLM:
  packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
  }

  # Weight name mapping from HF format to our format
  weight_name_mapping = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
  }

  def __init__(self, config: Qwen3Config):
    self.max_context = config.max_position_embeddings
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.blk = [Qwen3Block(config) for _ in range(config.num_hidden_layers)]
    self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    if config.tie_word_embeddings:
      self.output.weight = self.embed_tokens.weight

  def forward(self, tokens: Tensor, start_pos: int | UOp) -> Tensor:
    x = self.embed_tokens(tokens)
    for block in self.blk:
      x = block(x, start_pos)
    return self.output(self.output_norm(x))

  def __call__(self, tokens: Tensor, start_pos: int | UOp) -> Tensor:
    return self.forward(tokens, start_pos)
