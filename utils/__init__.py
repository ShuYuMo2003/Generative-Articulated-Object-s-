import torch

def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.to('cuda')
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(v) for v in obj]
    elif isinstance(obj, tuple):
        return (to_cuda(v) for v in obj)
    else:
        return obj

import random
import string

def generate_random_string(length):
    characters = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string