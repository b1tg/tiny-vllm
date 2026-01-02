from tinygrad import Tensor


def divide(numerator, denominator):
  assert numerator % denominator == 0
  return numerator // denominator


class LinearBase:

  def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None):
    self.tp_dim = tp_dim
    self.tp_rank = 0  # For single-GPU, always rank 0
    self.tp_size = 1  # For single-GPU
    self.weight = Tensor.empty(output_size, input_size)
    self.bias = Tensor.empty(output_size) if bias else None

  def __call__(self, x: Tensor) -> Tensor:
    raise NotImplementedError


class ReplicatedLinear(LinearBase):

  def __init__(self, input_size: int, output_size: int, bias: bool = False):
    super().__init__(input_size, output_size, bias)
    self.weight.weight_loader = self._weight_loader_impl
    if self.bias is not None:
      self.bias.weight_loader = self._weight_loader_impl

  def _weight_loader_impl(self, param: Tensor, loaded_weight: Tensor):
    param.assign(loaded_weight)

  def __call__(self, x: Tensor) -> Tensor:
    y = x.linear(self.weight.T)
    if self.bias is not None:
      y = y + self.bias
    return y


class ColumnParallelLinear(LinearBase):

  def __init__(self, input_size: int, output_size: int, bias: bool = False):
    tp_size = 1  # Single-GPU
    super().__init__(input_size, divide(output_size, tp_size), bias, 0)
    self.weight.weight_loader = self._weight_loader_impl
    if self.bias is not None:
      self.bias.weight_loader = self._weight_loader_impl

  def _weight_loader_impl(self, param: Tensor, loaded_weight: Tensor):
    param_data = param
    shard_size = param_data.shape[self.tp_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.shrink(((start_idx, start_idx + shard_size), None) if self.tp_dim == 0 else (None, (start_idx, start_idx + shard_size)))
    param_data.assign(loaded_weight)

  def __call__(self, x: Tensor) -> Tensor:
    y = x.linear(self.weight.T)
    if self.bias is not None:
      y = y + self.bias
    return y


class MergedColumnParallelLinear(ColumnParallelLinear):

  def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
    self.output_sizes = output_sizes
    self._loaded_weights = {}  # Track loaded weights
    super().__init__(input_size, sum(output_sizes), bias)
    # Attach weight_loader to the weight tensor
    self.weight.weight_loader = self._weight_loader_impl

  def _weight_loader_impl(self, param: Tensor, loaded_weight: Tensor, loaded_shard_id: int):
    # Store the loaded weight for this shard
    self._loaded_weights[loaded_shard_id] = loaded_weight

    # Check if all shards have been loaded
    if len(self._loaded_weights) == len(self.output_sizes):
      # Concatenate all weights in order
      weights_list = [self._loaded_weights[i] for i in range(len(self.output_sizes))]
      full_weight = weights_list[0].cat(*weights_list[1:], dim=0)
      param.assign(full_weight)
      self._loaded_weights.clear()


class QKVParallelLinear(ColumnParallelLinear):

  def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: int | None = None, bias: bool = False):
    tp_size = 1  # Single-GPU
    total_num_kv_heads = total_num_kv_heads or total_num_heads
    self.head_size = head_size
    self.num_heads = divide(total_num_heads, tp_size)
    self.num_kv_heads = divide(total_num_kv_heads, tp_size)
    output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
    self._loaded_weights = {}  # Track loaded q, k, v weights
    super().__init__(hidden_size, output_size, bias)
    # Attach weight_loader to the weight tensor
    self.weight.weight_loader = self._weight_loader_impl

  def _weight_loader_impl(self, param: Tensor, loaded_weight: Tensor, loaded_shard_id: str):
    assert loaded_shard_id in ["q", "k", "v"]
    # Store the loaded weight
    self._loaded_weights[loaded_shard_id] = loaded_weight

    # Check if all three (q, k, v) have been loaded
    if len(self._loaded_weights) == 3:
      # Concatenate in order: q, k, v
      full_weight = self._loaded_weights["q"].cat(self._loaded_weights["k"], self._loaded_weights["v"], dim=0)
      param.assign(full_weight)
      self._loaded_weights.clear()


class RowParallelLinear(LinearBase):

  def __init__(self, input_size: int, output_size: int, bias: bool = False):
    tp_size = 1  # Single-GPU
    super().__init__(divide(input_size, tp_size), output_size, bias, 1)
    self.weight.weight_loader = self._weight_loader_impl
    if self.bias is not None:
      self.bias.weight_loader = self._weight_loader_impl

  def _weight_loader_impl(self, param: Tensor, loaded_weight: Tensor):
    param_data = param
    shard_size = param_data.shape[self.tp_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.shrink((None, (start_idx, start_idx + shard_size)) if self.tp_dim == 1 else ((start_idx, start_idx + shard_size), None))
    param_data.assign(loaded_weight)

  def __call__(self, x: Tensor) -> Tensor:
    y = x.linear(self.weight.T)
    if self.bias is not None and self.tp_rank == 0:
      y = y + self.bias
    # For multi-GPU: dist.all_reduce(y)
    return y
