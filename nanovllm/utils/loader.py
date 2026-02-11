import os
from glob import glob
import numpy as np
from tinygrad import Tensor, nn
from safetensors import safe_open


def _convert_weight(loaded_weight_tensor):
  if hasattr(loaded_weight_tensor, 'dtype') and str(loaded_weight_tensor.dtype) == 'torch.bfloat16':
    np_weight = loaded_weight_tensor.to(dtype=__import__('torch').float16).numpy()
  elif hasattr(loaded_weight_tensor, 'numpy'):
    np_weight = loaded_weight_tensor.numpy()
  else:
    np_weight = np.array(loaded_weight_tensor)
  return Tensor(np_weight)


def _map_weight_name(name: str, packed_modules_mapping: dict) -> tuple[str, str | int | None]:
  """Map HF weight name to our model's weight name. Returns (mapped_name, shard_id)."""
  # Check if this is a packed module (q_proj, k_proj, v_proj, gate_proj, up_proj)
  for k, (v, shard_id) in packed_modules_mapping.items():
    if k in name:
      mapped = name.replace(k, v)
      mapped = _map_path(mapped)
      return mapped, shard_id

  return _map_path(name), None


def _map_path(name: str) -> str:
  """Map HF-style path to our model's path."""
  # model.layers.N.self_attn.X -> blk.N.X
  # model.layers.N.mlp.X -> blk.N.X
  # model.layers.N.input_layernorm -> blk.N.attn_norm
  # model.layers.N.post_attention_layernorm -> blk.N.ffn_norm
  # model.embed_tokens -> embed_tokens
  # model.norm -> output_norm
  # lm_head -> output

  name = name.replace("model.layers.", "blk.")
  name = name.replace(".self_attn.", ".")
  name = name.replace(".mlp.", ".")
  name = name.replace(".o_proj.", ".attn_output.")
  name = name.replace(".down_proj.", ".ffn_down.")
  name = name.replace(".input_layernorm.", ".attn_norm.")
  name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
  name = name.replace("model.embed_tokens.", "embed_tokens.")
  name = name.replace("model.norm.", "output_norm.")
  name = name.replace("lm_head.", "output.")

  # Handle q_norm, k_norm
  name = name.replace(".q_norm.", ".attn_q_norm.")
  name = name.replace(".k_norm.", ".attn_k_norm.")

  # Rename packed modules
  name = name.replace(".qkv_proj.", ".attn_qkv.")
  name = name.replace(".gate_up_proj.", ".ffn_gate_up.")

  return name


def load_model(model, path: str):
  """Load model weights from safetensors files."""
  packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

  for file in glob(os.path.join(path, "*.safetensors")):
    with safe_open(file, "pt", "cpu") as f:
      for weight_name in f.keys():
        mapped_name, shard_id = _map_weight_name(weight_name, packed_modules_mapping)

        param = _get_parameter(model, mapped_name)
        loaded_weight = _convert_weight(f.get_tensor(weight_name))

        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader and shard_id is not None:
          if param.dtype != loaded_weight.dtype:
            loaded_weight = loaded_weight.cast(param.dtype)
          weight_loader(param, loaded_weight, shard_id)
        else:
          if param.dtype != loaded_weight.dtype:
            loaded_weight = loaded_weight.cast(param.dtype)
          param.assign(loaded_weight)

  # Realize all parameters
  params = nn.state.get_parameters(model)
  for p in params:
    p.replace(p.contiguous())
  Tensor.realize(*params)


def _get_parameter(model, param_name: str):
  """Get a parameter from the model by name (dot-separated path)."""
  parts = param_name.split('.')
  obj = model
  for part in parts:
    if hasattr(obj, part):
      obj = getattr(obj, part)
    elif isinstance(obj, list):
      obj = obj[int(part)]
    else:
      raise AttributeError(f"Cannot find '{part}' in {type(obj)} (full path: {param_name})")
  return obj
