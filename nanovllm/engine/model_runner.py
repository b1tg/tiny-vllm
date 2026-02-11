from tinygrad import Tensor, Device, dtypes, UOp, TinyJit, getenv

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


class ModelRunner:

  def __init__(self, config: Config):
    self.config = config
    hf_config = config.hf_config

    # Initialize num_kvcache_blocks for the block manager
    if config.num_kvcache_blocks == -1:
      config.num_kvcache_blocks = 1024

    # Initialize model
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    self.sampler = Sampler()

    # Symbolic variable for start_pos to enable kernel caching
    self.v_start_pos = UOp.variable("start_pos", 1, hf_config.max_position_embeddings - 1)

    # JIT for decode path
    self.forward_jit = TinyJit(self.model.forward)

  def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    if is_prefill:
      return self._run_prefill(seqs)
    else:
      return self._run_decode(seqs)

  def _run_prefill(self, seqs: list[Sequence]) -> list[int]:
    # Flatten all input tokens
    input_ids = []
    for seq in seqs:
      input_ids.extend(seq[seq.num_cached_tokens:])

    tokens = Tensor([input_ids], dtype=dtypes.int32)  # (1, T)
    start_pos = 0

    # Prefill: use regular forward (not JIT'd since T > 1)
    logits = self.model(tokens, start_pos)  # (1, T, vocab_size)

    # Sample from last token's logits for each sequence
    # For simplicity, take the last logit position
    logits = logits[:, -1:, :]  # (1, 1, vocab_size)

    temperatures = Tensor([seq.temperature for seq in seqs], dtype=dtypes.float32)
    sample_tokens = self.sampler(logits.squeeze(0), temperatures)
    token_ids = sample_tokens.tolist()

    # Store the total prompt length for decode
    self._start_pos = len(input_ids)

    return token_ids

  def _run_decode(self, seqs: list[Sequence]) -> list[int]:
    input_ids = [seq.last_token for seq in seqs]
    tokens = Tensor([input_ids], dtype=dtypes.int32)  # (1, 1)

    start_pos = self._start_pos
    # Use symbolic start_pos for JIT
    sym_start_pos = self.v_start_pos.bind(start_pos)

    # Decode: use JIT'd forward
    logits = (self.forward_jit if getenv("JIT", 1) else self.model.forward)(tokens, sym_start_pos)  # (1, 1, vocab_size)

    temperatures = Tensor([seq.temperature for seq in seqs], dtype=dtypes.float32)
    sample_tokens = self.sampler(logits.squeeze(0), temperatures)
    token_ids = sample_tokens.tolist()

    self._start_pos = start_pos + 1

    return token_ids
