from transformer.utils import parse_args

def get_decoder(config):
    type = config['decoder']['type']
    if type == 'NativeDecoder':
        from transformer.model.decoder import NativeDecoder
        return NativeDecoder(**parse_args(config, config['decoder']['args']))
    elif type == 'ParallelDecoder':
        from transformer.model.decoder import ParallelDecoder
        return ParallelDecoder(**parse_args(config, config['decoder']['args']))
    elif type == 'DecoderV2':
        from transformer.model.decoder import DecoderV2
        return DecoderV2(**parse_args(config, config['decoder']['args']))
    else:
        raise NotImplementedError(f"{type} is not implemented")