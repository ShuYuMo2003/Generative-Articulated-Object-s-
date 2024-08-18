import torch
import requests
from PIL import Image
from rich import print
from pathlib import Path
from transformers import Blip2Processor, Blip2Model

import sys
sys.path.append('../')
from utils.logging import Log, console

blip_cache_path = Path('../cache/blip_cache')
blip_model_name = "Salesforce/blip2-opt-2.7b"

if __name__ == '__main__':
    blip_cache_path.mkdir(parents=True, exist_ok=True)
    blip_device = "cuda" if torch.cuda.is_available() else "cpu"

    blip_processor = Blip2Processor.from_pretrained(blip_model_name, cache_dir=blip_cache_path.as_posix())
    model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=blip_cache_path.as_posix(), torch_dtype=torch.float16).to(blip_device)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = blip_processor(images=image, return_tensors="pt").to(blip_device, torch.float16)
    output = model.get_image_features(**inputs)
    image_feature = output.last_hidden_state

    print(type(inputs.pixel_values), inputs.pixel_values.shape)
    print(type(image_feature), image_feature.shape)


