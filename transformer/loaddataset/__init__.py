from transformer.utils import parse_args

def get_dataset(config, **args):
    dataset = config['dataset']['type']
    if dataset == 'redis':
        from .redis import FromRedisNativeDataset
        return FromRedisNativeDataset(**config['dataset']['args'])
    elif dataset == 'redis_parallel':
        from .redis_parallel import FromRedisParallelDataset
        return FromRedisParallelDataset(part_structure=config['part_structure'], **config['dataset']['args'])
    else:
        raise NotImplementedError(f"{dataset}: dataset is not implemented")