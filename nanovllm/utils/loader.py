import os
from glob import glob
import numpy as np
from tinygrad import Tensor
from safetensors import safe_open


def default_weight_loader(param: Tensor, loaded_weight: Tensor):
  param.assign(loaded_weight)


def load_model(model, path: str):
  """Load model weights from safetensors files."""
  packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

  for file in glob(os.path.join(path, "*.safetensors")):
    with safe_open(file, "pt", "cpu") as f:
      for weight_name in f.keys():
        # Check if this weight is part of a packed module
        for k in packed_modules_mapping:
          if k in weight_name:
            v, shard_id = packed_modules_mapping[k]
            param_name = weight_name.replace(k, v)
            param = get_parameter(model, param_name)
            weight_loader = getattr(param, "weight_loader", None)
            loaded_weight_tensor = f.get_tensor(weight_name)
            # Convert torch tensor to numpy - handle bfloat16
            if hasattr(loaded_weight_tensor, 'dtype') and str(loaded_weight_tensor.dtype) == 'torch.bfloat16':
              np_weight = loaded_weight_tensor.float().numpy()
            elif hasattr(loaded_weight_tensor, 'numpy'):
              np_weight = loaded_weight_tensor.numpy()
            else:
              np_weight = np.array(loaded_weight_tensor)
            loaded_weight = Tensor(np_weight)
            if weight_loader:
              weight_loader(param, loaded_weight, shard_id)
            else:
              param.assign(loaded_weight)
            break
        else:
          # Not a packed module
          param = get_parameter(model, weight_name)
          weight_loader = getattr(param, "weight_loader", default_weight_loader)
          loaded_weight_tensor = f.get_tensor(weight_name)
          # Convert torch tensor to numpy - handle bfloat16
          if hasattr(loaded_weight_tensor, 'dtype') and str(loaded_weight_tensor.dtype) == 'torch.bfloat16':
            np_weight = loaded_weight_tensor.float().numpy()
          elif hasattr(loaded_weight_tensor, 'numpy'):
            np_weight = loaded_weight_tensor.numpy()
          else:
            np_weight = np.array(loaded_weight_tensor)
          loaded_weight = Tensor(np_weight)
          weight_loader(param, loaded_weight)


def get_parameter(model, param_name: str):
  """Get a parameter from the model by name (dot-separated path)."""
  parts = param_name.split('.')
  obj = model
  for part in parts:
    if hasattr(obj, part):
      obj = getattr(obj, part)
    elif isinstance(obj, list):
      obj = obj[int(part)]
    else:
      raise AttributeError(f"Cannot find {part} in {type(obj)}")
  return obj
