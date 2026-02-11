from tinygrad import Tensor, UOp, nn


class RMSNorm:

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    self.eps = eps
    self.weight = Tensor.ones(hidden_size)

  def __call__(self, x: Tensor) -> Tensor:
    return (x * (x.float().square().mean(axis=-1, keepdim=True) + self.eps).rsqrt()).cast(x.dtype) * self.weight
