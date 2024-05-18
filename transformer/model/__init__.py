def get_decoder(config):
    type = config['type']
    if type == 'NativeDecoder':
        from transformer.model.decoder import NativeDecoder
        return NativeDecoder(**config['args'])
    else:
        raise NotImplementedError(f"{type} is not implemented")