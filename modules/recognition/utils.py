from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import math
from PIL import Image
from configs.recognition_config import image_height, image_max_width, image_min_width
import albumentations as A

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    
    return new_w, expected_height

transform = transforms.Compose([
    transforms.ToTensor(), # /255
])

aug = A.Compose(
    [
        A.InvertImg(p=0.2),
        A.ColorJitter(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Perspective(scale=(0.01, 0.05)),
    ]
)

def process_image(image, image_height=image_height, image_min_width=image_min_width, image_max_width=image_max_width):
    img = image.convert("RGB")
    # resize
    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    img = img.resize((new_w, image_height), Image.LANCZOS)
    # scale to 0-1
    img = transform(img)
    return img
    