import os
import time
from rich import print
from glob import glob
from pathlib import Path
from sensitive_info import *
from tqdm import tqdm
# https://github.com/snowby666/poe-api-wrapper
from poe_api_wrapper import PoeApi

client = PoeApi(cookie=poe_tokens)
bot_type = "gpt-4o"
chatcode = "2de6a90dp9gehmhmd4t"
figures = list(glob('dataset/4_screenshot/*'))
prompt = ('这是一个 CATE，请着重描述一下他的形状特征和铰接（各部分的运动）特征，'
         +'忽略颜色和纹理特征，给出简短的一句话描述。请使用英文回答问题。')

description_output_path = 'dataset/4_screenshot_description/'
Path(description_output_path).mkdir(exist_ok=True)

for figure in tqdm(figures):
    output_path = os.path.join(description_output_path, figure.split('/')[-1].replace('.png', '.txt'))
    abs_path = os.path.abspath(figure)
    category = figure.split('/')[-1].split('-')[0]
    print('describing fig:', abs_path, 'with', category)
    description = ''
    while True:
        try:
            for chunk in client.send_message(bot_type, prompt.replace('CATE', category), chatCode="2ddygyofk1z97yo3vop", file_path=[abs_path]):
                print(chunk["response"], end="", flush=True)
                description += chunk["response"]
            break
        except Exception as e:
            if 'You have sent messages too fast' in str(e):
                print('You have sent messages too fast, please wait for a while.')
            time.sleep(2)

    print('')
    with open(output_path, 'w') as f:
        f.write(description)
    print('')
