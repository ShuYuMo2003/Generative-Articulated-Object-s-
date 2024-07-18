import os
import time
from rich import print
from glob import glob
from pathlib import Path
from sensitive_info import *
from tqdm import tqdm
# https://github.com/snowby666/poe-api-wrapper
from poe_api_wrapper import PoeApi

client = PoeApi(tokens=poe_tokens)
bot_type = "gpt4_o"
# chatcode = "2de6a90dp9gehmhmd4t"
# figures = list(glob('dataset/4_screenshot/Toilet_2701_102701/Toilet_2701_102701_0.png'))
figures = list(glob('dataset/4_screenshot/USB*/*0.png'))


# prompt = ('这是一个 CATE，请着重描述一下他的形状特征和铰接（各部分的运动）特征，'
#          +'忽略颜色和纹理特征，给出简短的一句话描述。请使用英文回答问题。')
# prompt = ("This is a CATE. "
#          +"Please focus on describing its shape features and articulation characteristics, "
#          +"mainly referring to the movement features of its parts. "
#          +"Ignore color and texture features, "
#          +"and provide a brief one-sentence description.")

prompt_path = 'scripts/description_prompt.txt'
if not os.path.exists(prompt_path):
    raise FileNotFoundError(prompt_path + ' not found')
with open(prompt_path, 'r') as f:
    prompt = f.read()

description_output_path = 'dataset/4_screenshot_description/'
Path(description_output_path).mkdir(exist_ok=True)

# print(prompt)

for figure in tqdm(figures):
    output_path = os.path.join(description_output_path, figure.split('/')[-1].replace('.png', '.txt'))
    abs_path = os.path.abspath(figure)
    category = figure.split('/')[-1].split('_')[0]

    if os.path.exists(output_path):
        print('file exist:', output_path)
        continue

    print('describing fig:', abs_path, 'with', category)
    description = ''
    if not os.path.exists(abs_path):
        print('file not exist:', abs_path)
    for chunk in client.send_message(bot_type, prompt.replace('CATE', category), file_path=[abs_path]):
        print(chunk["response"], end="", flush=True)
        description += chunk["response"]
    # while True:
        # try:
        #     for chunk in client.send_message(bot_type, prompt.replace('CATE', category), chatCode="2ddygyofk1z97yo3vop", file_path=[abs_path]):
        #         print(chunk["response"], end="", flush=True)
        #         description += chunk["response"]
        #     break
        # except Exception as e:
        #     if 'You have sent messages too fast' in str(e):
        #         print('You have sent messages too fast, please wait for a while.')
        #     time.sleep(2)

    print('')
    with open(output_path, 'w') as f:
        f.write(description)
    print('')
