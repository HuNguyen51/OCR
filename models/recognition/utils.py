from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import math
from PIL import Image
from configs.recognition_config import image_height, image_max_width, image_min_width

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    
    return new_w, expected_height

transform = transforms.Compose([
    # transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def process_image(image, image_height=image_height, image_min_width=image_min_width, image_max_width=image_max_width):
    img = image.convert("RGB")

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    result = Image.new(img.mode, (image_max_width, image_height), (0,0,0))
    result.paste(img, (0, 0))

    # new_w: từ đây trở đi sẽ là image padding mask
    
    return transform(result) #, image_padding_mask

# Hàm collate_fn để xử lý batch dữ liệu
def padding(text_ids, maxlen=128):
    if len(text_ids) < maxlen:
        text_ids += [0]*(maxlen-len(text_ids))
    else:
        text_ids = text_ids[:maxlen]
    return text_ids
    