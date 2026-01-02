from tinygrad import Tensor, Device, dtypes

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

  def __init__(self, config: Config):
    self.config = config
    hf_config = config.hf_config
    self.block_size = config.kvcache_block_size
    self.enforce_eager = config.enforce_eager

    # Set default device - use CPU for now to avoid CUDA issues
    Device.DEFAULT = "CPU"
    # Device.DEFAULT = "AMD"

    # Set dtype based on config - use float32 since we convert bfloat16 weights to float32
    dtype_map = {
      "float32": dtypes.float32,
      "float16": dtypes.float16,
      "bfloat16": dtypes.float32,  # Convert bfloat16 to float32
    }
    self.dtype = dtype_map.get(str(hf_config.torch_dtype).split('.')[-1], dtypes.float32)

    # Initialize model
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    self.sampler = Sampler()

    # Allocate KV cache
    self.allocate_kv_cache()

  def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config

    # Estimate memory usage and allocate KV cache
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

    # For single-GPU, simplified allocation
    # Assume we have enough memory for the configured number of blocks
    if config.num_kvcache_blocks == -1:
      config.num_kvcache_blocks = 1024  # Default value

    # KV cache shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    self.kv_cache = Tensor.zeros(
      2,
      hf_config.num_hidden_layers,
      config.num_kvcache_blocks,
      self.block_size,
      num_kv_heads,
      head_dim,
      dtype=self.dtype
    )

    # Assign k_cache and v_cache to each attention layer
    layer_id = 0
    for layer in self.model.model.layers:
      if hasattr(layer.self_attn, 'attn'):
        layer.self_attn.attn.k_cache = self.kv_cache[0, layer_id]
        layer.self_attn.attn.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1

  def prepare_block_tables(self, seqs: list[Sequence]):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = Tensor(block_tables, dtype=dtypes.int32)
    return block_tables

  def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None

    for seq in seqs:
      seqlen = len(seq)
      input_ids.extend(seq[seq.num_cached_tokens:])
      positions.extend(list(range(seq.num_cached_tokens, seqlen)))
      seqlen_q = seqlen - seq.num_cached_tokens
      seqlen_k = seqlen
      cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
      cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
      max_seqlen_q = max(seqlen_q, max_seqlen_q)
      max_seqlen_k = max(seqlen_k, max_seqlen_k)

      if not seq.block_table:
        continue

      for i in range(seq.num_cached_blocks, seq.num_blocks):
        start = seq.block_table[i] * self.block_size
        if i != seq.num_blocks - 1:
          end = start + self.block_size
        else:
          end = start + seq.last_block_num_tokens
        slot_mapping.extend(list(range(start, end)))

    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
      block_tables = self.prepare_block_tables(seqs)

    input_ids = Tensor(input_ids, dtype=dtypes.int64)
    positions = Tensor(positions, dtype=dtypes.int64)
    cu_seqlens_q = Tensor(cu_seqlens_q, dtype=dtypes.int32)
    cu_seqlens_k = Tensor(cu_seqlens_k, dtype=dtypes.int32)
    slot_mapping = Tensor(slot_mapping, dtype=dtypes.int32)

    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    return input_ids, positions

  def prepare_decode(self, seqs: list[Sequence]):
    # TEMPORARY: Recompute full sequence on each step (slow but correct without KV cache)
    # This treats decode like prefill - reprocesses all tokens
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0

    for seq in seqs:
      seqlen = len(seq)
      input_ids.extend(seq.token_ids)  # All tokens, not just last
      positions.extend(list(range(seqlen)))
      cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
      cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen)
      max_seqlen_q = max(seqlen, max_seqlen_q)
      max_seqlen_k = max(seqlen, max_seqlen_k)

    input_ids = Tensor(input_ids, dtype=dtypes.int64)
    positions = Tensor(positions, dtype=dtypes.int64)
    cu_seqlens_q = Tensor(cu_seqlens_q, dtype=dtypes.int32)
    cu_seqlens_k = Tensor(cu_seqlens_k, dtype=dtypes.int32)
    slot_mapping = Tensor([], dtype=dtypes.int32)

    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, None)
    return input_ids, positions

  def prepare_sample(self, seqs: list[Sequence]):
    temperatures = [seq.temperature for seq in seqs]
    temperatures = Tensor(temperatures, dtype=dtypes.float32)
    return temperatures

  def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs)

    # Run model
    hidden_states = self.model(input_ids, positions)
    logits = self.model.compute_logits(hidden_states)

    # Sample tokens
    sample_tokens = self.sampler(logits, temperatures)

    # Convert to list
    token_ids = sample_tokens.numpy().tolist() if hasattr(sample_tokens, 'numpy') else sample_tokens.tolist()

    reset_context()
    return token_ids
