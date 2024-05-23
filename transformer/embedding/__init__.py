def get_tokenizer(config, **add_args):
    type = config['tokenizer']['type']
    if type == 'NativeMLPTokenizer':
        from transformer.embedding.tokenizer import NativeMLPTokenizer
        return NativeMLPTokenizer(config=config, **add_args, **config['tokenizer']['args'])
    else:
        raise NotImplementedError(f"{type} is not implemented")