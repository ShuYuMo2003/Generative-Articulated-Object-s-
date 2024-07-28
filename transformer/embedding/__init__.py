from rich import print
from transformer.utils import parse_args

def get_tokenizer(config):
    type = config['tokenizer']['type']
    if type == 'NativeMLPTokenizer':
        from transformer.embedding.tokenizer import NativeMLPTokenizer
        # print(parse_args(config, config['tokenizer']['args']))
        return NativeMLPTokenizer(**parse_args(config, config['tokenizer']['args']))
    elif type == 'MLPTokenizerV2':
        from transformer.embedding.tokenizer import MLPTokenizerV2
        return MLPTokenizerV2(**parse_args(config, config['tokenizer']['args']))
    else:
        raise NotImplementedError(f"{type} is not implemented")

def get_positionembedding(config):
    type = config['position_embedding']['type']
    if type == 'NativeCatPositionEmbedding':
        from transformer.embedding.position import NativeCatPositionEmbedding
        return NativeCatPositionEmbedding(**parse_args(config, config['position_embedding']['args']))
    elif type == 'PositionGRUEmbedding':
        from transformer.embedding.position import PositionGRUEmbedding
        return PositionGRUEmbedding(**parse_args(config, config['position_embedding']['args']))
    else:
        raise NotImplementedError(f"{type} is not implemented")

def get_g_embedding(config):
    type = config['g_embedding']['type']
    if type == 'NativeGEmbedding':
        from transformer.embedding.g_embedding import NativeGEmbedding
        return NativeGEmbedding(**parse_args(config, config['g_embedding']['args']))
    else:
        raise NotImplementedError(f"{type} is not implemented")

def get_untokenizer(config):
    type = config['untokenizer']['type']
    if type == 'NativeMLPUnTokenizer':
        from transformer.embedding.untokenizer import NativeMLPUnTokenizer
        return NativeMLPUnTokenizer(**parse_args(config, config['untokenizer']['args']))
    elif type == 'MLPUnTokenizerV2':
        from transformer.embedding.untokenizer import MLPUnTokenizerV2
        return MLPUnTokenizerV2(**parse_args(config, config['untokenizer']['args']))
    else:
        raise NotImplementedError(f"{type} is not implemented")