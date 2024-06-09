import torch

def get_value_from_url(config, url):
    if url == '.': return config
    d = config
    for key in url.split('.'):
        try:
            d = d[key]
        except:
            raise ValueError(f"{url}: {key} not found.")
    return d

def parse_args(config, args):
    parsed_args = {}
    for key, url_to_value in args.items():
        parsed_args[key] = get_value_from_url(config, url_to_value)
    return parsed_args

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

def str2hash(ss):
    import hashlib
    return int(hashlib.md5(ss.encode()).hexdigest(), 16)

__all__ = ['parse_args', 'to_cuda']