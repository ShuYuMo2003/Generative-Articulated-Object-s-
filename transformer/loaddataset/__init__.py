from transformer.utils import parse_args

def get_dataset(config, train):
    if train:
        dataset = config['dataset']['type']
        args = config['dataset']['args']
    else:
        dataset = config['evaluate_dataset']['type']
        args = config['evaluate_dataset']['args']

    if dataset == 'redis':
        from .redis import FromRedisNativeDataset
        return FromRedisNativeDataset(**args)
    elif dataset == 'redis_parallel':
        from .redis_parallel import FromRedisParallelDataset
        return FromRedisParallelDataset(part_structure=config['part_structure'], **args)
    elif dataset == 'file':
        from .filesdataset import FileSysDataset
        return FileSysDataset(**args)
    else:
        raise NotImplementedError(f"{dataset}: dataset is not implemented")