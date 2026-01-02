from tinygrad import Tensor


class RMSNorm:

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    self.eps = eps
    self.weight = Tensor.ones(hidden_size)

  def rms_forward(self, x: Tensor) -> Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = (x * x).mean(axis=-1, keepdim=True)
    x = x * (var + self.eps).rsqrt()
    x = x.cast(orig_dtype) * self.weight
    return x

  def add_rms_forward(self, x: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
    orig_dtype = x.dtype
    x = x.float() + residual.float()
    residual = x.cast(orig_dtype)
    var = (x * x).mean(axis=-1, keepdim=True)
    x = x * (var + self.eps).rsqrt()
    x = x.cast(orig_dtype) * self.weight
    return x, residual

  def __call__(self, x: Tensor, residual: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
    if residual is None:
      return self.rms_forward(x)
    else:
      return self.add_rms_forward(x, residual)
