import os
import re
import time
from rich import print
from glob import glob
from pathlib import Path
from tqdm import tqdm
# https://github.com/snowby666/poe-api-wrapper
from poe_api_wrapper import PoeApi

import sys
sys.path.append('..')
from sensitive_info import poe_tokens

client = PoeApi(tokens=poe_tokens)
bot_type = "gpt4_o"

prompt =  \
'''This is a type of CATE.
Please focus on its movable parts and articulation characteristics, and describe the possible motion characteristics of each part.
In the given image, there are different colored parts that can move relative to each other.
In your description, you should ignore the color, texture, and other non-structural features.
Provide a brief one-sentence description.
Here are some examples:
---
input: <image of a USB>
description: A USB device features a rectangular shape with a swivel mechanism that allows a cover to rotate and reveal or protect the USB connector.
--
input: <image of a USB>
description: A USB device features a rectangular shape with a removable cap that detaches to expose the USB connector.
---

For more complex shapes, you can describe the motion characteristics of the main part in detail with more sentences.
'''


description_output_path = '../dataset/4_screenshot_description/'
Path(description_output_path).mkdir(exist_ok=True)

def camel_to_snake(name):
    # StorageFurniture -> storage furniture
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def compress_figure(figure_path: str):
    return figure_path
    pass


def generate_description(figure_path, output_txt_path, category):
    print('describing fig:', figure_path, 'with', category, 'writing to', output_txt_path)

    if os.path.exists(output_txt_path) and "Message too long" in Path(output_txt_path).read_text():
        os.remove(output_txt_path)
        print('[Delete]: ', output_txt_path)

    if os.path.exists(output_txt_path):
        print('description for', figure_path, 'already exists.')
        return

    print('describing fig:', figure_path, 'with', category)
    while True:
        try:
            description = ''
            for chunk in client.send_message(
                                bot_type,
                                prompt.replace('CATE', category),
                                file_path=[figure_path]):
                print(chunk["response"], end="", flush=True)
                description += chunk["response"]
            # print(description)
            if "Message too long" in description:
                print('[Error] Message too long')
                figure_path = compress_figure(figure_path)
                raise Exception('Message too long')
            break
        except Exception as e:
            print(e)
            time.sleep(2)

    print('[Write] ', output_txt_path, ": ", description)
    with open(output_txt_path, 'w') as f:
        f.write(description)

if __name__ == '__main__':
    screenshot_paths = glob('../dataset/4_screenshot/*/*.png')
    for screenshot_path in tqdm(screenshot_paths):
        output_path = screenshot_path.replace('4_screenshot', '4_screenshot_description')   \
                                     .replace('.png', '.txt')
        Path(output_path).parent.mkdir(exist_ok=True)
        shape_name = Path(screenshot_path).parent.name
        category = shape_name.split('_')[0]
        category = camel_to_snake(category)
        generate_description(screenshot_path, output_path, category)