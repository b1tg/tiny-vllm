from tinygrad import Tensor
from nanovllm.utils.context import get_context


class VocabParallelEmbedding:

  def __init__(self, num_embeddings: int, embedding_dim: int):
    self.tp_rank = 0  # Single-GPU
    self.tp_size = 1  # Single-GPU
    assert num_embeddings % self.tp_size == 0
    self.num_embeddings = num_embeddings
    self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
    self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
    self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
    self.weight = Tensor.empty(self.num_embeddings_per_partition, embedding_dim)
    self.weight.weight_loader = self._weight_loader_impl

  def _weight_loader_impl(self, param: Tensor, loaded_weight: Tensor):
    param_data = param
    shard_size = param_data.shape[0]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.shrink(((start_idx, start_idx + shard_size), None))
    if param_data.dtype != loaded_weight.dtype:
      loaded_weight = loaded_weight.cast(param_data.dtype)
    param_data.assign(loaded_weight)

  def __call__(self, x: Tensor):
    if self.tp_size > 1:
      mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
      x = mask * (x - self.vocab_start_idx)
    y = self.weight[x]
    if self.tp_size > 1:
      y = mask.unsqueeze(1) * y
      # dist.all_reduce(y)
    return y


class ParallelLMHead(VocabParallelEmbedding):

  def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
    assert not bias
    super().__init__(num_embeddings, embedding_dim)

  def __call__(self, x: Tensor):
    context = get_context()
    if context.is_prefill:
      last_indices = context.cu_seqlens_q[1:] - 1
      x = x[last_indices]
    logits = x.linear(self.weight.T)
    if self.tp_size > 1:
      # For multi-GPU: gather and concatenate
      pass
    return logits
