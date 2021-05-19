def torch_device_gpu_if_available():
  import torch
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
