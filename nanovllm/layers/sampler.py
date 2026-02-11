from tinygrad import Tensor


class Sampler:

  def __call__(self, logits: Tensor, temperatures: Tensor):
    logits = logits.float() / temperatures.unsqueeze(dim=1)
    probs = logits.softmax(axis=-1)
    # Gumbel-max sampling
    uniform = Tensor.rand(*probs.shape)
    gumbel = -(-(uniform + 1e-10).log()).log()
    sample_tokens = (logits + gumbel).argmax(axis=-1)
    return sample_tokens
