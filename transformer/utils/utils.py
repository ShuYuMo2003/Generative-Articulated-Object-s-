import torch

def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.to('cuda')
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(v) for v in obj]
    else:
        return obj