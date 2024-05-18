def get_tokenizer(config):
    type = config['tokenizer']['type']
    if type == 'NativeMLPTokenizer':
        from transformer.embedding.tokenizer import NativeMLPTokenizer
        return NativeMLPTokenizer(**config['tokenizer']['args'])
    else:
        raise NotImplementedError(f"{type} is not implemented")