import yaml
import torch
import numpy as np
import random
import argparse
from rich import print

from transformer.utils import str2hash

from onet import a_extract_from_raw_dataset as process_raw_dataset
from onet import b_convert_to_onet_dataset as generate_onet_dataset
from onet import d_evaluate_latent_code as gen_latent_code
from onet import e_push_to_redis as push_to_redis

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('yaml configuration files to use'),
                    type=argparse.FileType('r'), required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read())
setup_seed(str2hash(config['random_seed']) & ((1 << 20) - 1))
config.update({
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

actions = config['action']

print(config)
print('=================', 'run action:', str(actions))

for (step_idx, action) in enumerate(actions):
    print('START ======================== step = ', step_idx, '  running: ', action, '========================')
    if action == 'process_raw_dataset':
        raise RuntimeError('Deprecated. Use process_dataset 1.py instead.')
        exit(-1)
        process_raw_dataset.main(config)
    elif action == 'generate_onet_dataset':
        generate_onet_dataset.main(config)
    elif action == 'gen_latent_code':
        gen_latent_code.main(config)
    elif action == 'push_to_redis':
        push_to_redis.main(config)
    print('END ========================== step = ', step_idx, '  running: ', action, '========================')