from transformer.utils import parse_args

def get_decoder(config):
    type = config['decoder']['type']
    if type == 'NativeDecoder':
        from transformer.model.decoder import NativeDecoder
        return NativeDecoder(**parse_args(config, config['decoder']['args']))
    else:
        raise NotImplementedError(f"{type} is not implemented")