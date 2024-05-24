def get_dataset(config):
    dataset = config['dataset']['type']
    if dataset == 'redis':
        from .redis import FromRedisNativeDataset
        return FromRedisNativeDataset(**config['dataset']['args'])
    else:
        raise NotImplementedError(f"{dataset}: dataset is not implemented")