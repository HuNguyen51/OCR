from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from models.recognition.utils import padding
import random

# Định nghĩa lớp Dataset tùy chỉnh
class OCRVietnamese(Dataset):
  def __init__(self, data, vocab, transform=None, maxlen=128): 
    super(OCRVietnamese, self).__init__()
    self.data = data
    self.vocab = vocab
    self.transform = transform
    self.maxlen = maxlen

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    image = Image.open(self.data[idx]['image_path'])

    if self.transform:
      image = self.transform(image)

    target = [self.vocab.char2idx['<sos>']]
    for char in self.data[idx]['text']:
      if char in self.vocab.char2idx:
        target.append(self.vocab.char2idx[char])
      else:
        target.append(self.vocab.char2idx['<unk>'])
    target.append(self.vocab.char2idx['<eos>'])
    target = padding(target, self.maxlen)

    target = torch.tensor(target)

    return image, target, self.data[idx]['image_path']
  
  
  def get_random_image(self):
    idx = random.randint(0,len(self.data))
    path = self.data[idx]['image_path']

    image = Image.open(self.data[idx]['image_path'])
    if self.transform:
      image = self.transform(image)
    
    text = self.data[idx]['text']
    return image, text, path
    
