from tinygrad import Tensor


class SiluAndMul:

  def __call__(self, x: Tensor) -> Tensor:
    x, y = x.chunk(2, -1)
    return x.silu() * y
