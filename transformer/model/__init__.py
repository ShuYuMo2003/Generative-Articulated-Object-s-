def get_decoder(config):
    type = config['decoder']['type']
    if type == 'NativeDecoder':
        from transformer.model.decoder import NativeDecoder
        return NativeDecoder(config=config, **config['decoder']['args'])
    else:
        raise NotImplementedError(f"{type} is not implemented")