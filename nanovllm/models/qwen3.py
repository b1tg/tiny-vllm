from tinygrad import Tensor
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention:

  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_position: int = 4096 * 32,
    head_dim: int | None = None,
    rms_norm_eps: float = 1e-06,
    qkv_bias: bool = False,
    rope_theta: float = 10000,
    rope_scaling: tuple | None = None,
  ):
    tp_size = 1  # Single-GPU
    self.total_num_heads = num_heads
    assert self.total_num_heads % tp_size == 0
    self.num_heads = self.total_num_heads // tp_size
    self.total_num_kv_heads = num_kv_heads
    assert self.total_num_kv_heads % tp_size == 0
    self.num_kv_heads = self.total_num_kv_heads // tp_size
    self.head_dim = head_dim or hidden_size // self.total_num_heads
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim ** -0.5
    self.qkv_bias = qkv_bias

    self.qkv_proj = QKVParallelLinear(
      hidden_size,
      self.head_dim,
      self.total_num_heads,
      self.total_num_kv_heads,
      bias=qkv_bias,
    )
    self.o_proj = RowParallelLinear(
      self.total_num_heads * self.head_dim,
      hidden_size,
      bias=False,
    )
    self.rotary_emb = get_rope(
      self.head_dim,
      rotary_dim=self.head_dim,
      max_position=max_position,
      base=rope_theta,
      rope_scaling=rope_scaling,
    )
    self.attn = Attention(
      self.num_heads,
      self.head_dim,
      self.scaling,
      self.num_kv_heads,
    )
    if not self.qkv_bias:
      self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
      self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

  def __call__(self, positions: Tensor, hidden_states: Tensor) -> Tensor:
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.reshape(-1, self.num_heads, self.head_dim)
    k = k.reshape(-1, self.num_kv_heads, self.head_dim)
    v = v.reshape(-1, self.num_kv_heads, self.head_dim)
    if not self.qkv_bias:
      q = self.q_norm(q)
      k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)
    output = self.o_proj(o.flatten(1, -1))
    return output


class Qwen3MLP:

  def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
    self.gate_up_proj = MergedColumnParallelLinear(
      hidden_size,
      [intermediate_size] * 2,
      bias=False,
    )
    self.down_proj = RowParallelLinear(
      intermediate_size,
      hidden_size,
      bias=False,
    )
    assert hidden_act == "silu"
    self.act_fn = SiluAndMul()

  def __call__(self, x):
    gate_up = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x = self.down_proj(x)
    return x


class Qwen3DecoderLayer:

  def __init__(self, config: Qwen3Config):
    self.self_attn = Qwen3Attention(
      hidden_size=config.hidden_size,
      num_heads=config.num_attention_heads,
      num_kv_heads=config.num_key_value_heads,
      max_position=config.max_position_embeddings,
      rms_norm_eps=config.rms_norm_eps,
      qkv_bias=getattr(config, 'attention_bias', True),
      head_dim=getattr(config, 'head_dim', None),
      rope_theta=getattr(config, "rope_theta", 1000000),
      rope_scaling=getattr(config, "rope_scaling", None),
    )
    self.mlp = Qwen3MLP(
      hidden_size=config.hidden_size,
      intermediate_size=config.intermediate_size,
      hidden_act=config.hidden_act,
    )
    self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def __call__(self, positions: Tensor, hidden_states: Tensor, residual: Tensor | None) -> tuple[Tensor, Tensor]:
    if residual is None:
      hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
      hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


class Qwen3Model:

  def __init__(self, config: Qwen3Config):
    self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
    self.layers = [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def __call__(self, input_ids: Tensor, positions: Tensor) -> Tensor:
    hidden_states = self.embed_tokens(input_ids)
    residual = None
    for layer in self.layers:
      hidden_states, residual = layer(positions, hidden_states, residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


class Qwen3ForCausalLM:
  packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
  }

  def __init__(self, config: Qwen3Config):
    self.model = Qwen3Model(config)
    self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
    if config.tie_word_embeddings:
      self.lm_head.weight = self.model.embed_tokens.weight

  def __call__(self, input_ids: Tensor, positions: Tensor) -> Tensor:
    return self.model(input_ids, positions)

  def compute_logits(self, hidden_states: Tensor) -> Tensor:
    return self.lm_head(hidden_states)
