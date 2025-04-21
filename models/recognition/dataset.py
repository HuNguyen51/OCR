from torch.utils.data import Dataset
import torch
from PIL import Image
import random
from models.recognition.utils import resize
from configs.recognition_config import image_height, image_max_width, image_min_width
from tqdm.notebook import tqdm
from torch.utils.data.sampler import Sampler



# Định nghĩa lớp Dataset tùy chỉnh
class OCRVietnamese(Dataset):
  def __init__(self, data, vocab, transform=None, maxlen=128,
               image_height=image_height, image_max_width=image_max_width, image_min_width=image_min_width): 
    super(OCRVietnamese, self).__init__()
    self.data = data
    self.vocab = vocab
    self.transform = transform
    self.maxlen = maxlen

    self.image_height = image_height
    self.image_max_width = image_max_width
    self.image_min_width = image_min_width

    self.cluster_image = self.__create_cluster(data)

  def __create_cluster(self, data):
    # -> dict(new_w: (im_idx_1,...),)
    self.cluster_image = dict()
    for idx in tqdm(range(len(data))):
      image = Image.open(self.data[idx]['image_path'])
      w, h = image.size
      new_w, _ = resize(
            w, h, self.image_height, self.image_min_width, self.image_max_width
        )
      
      self.cluster_image.setdefault(new_w, [])
      self.cluster_image[new_w].append(idx)
    return self.cluster_image
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    image = Image.open(self.data[idx]['image_path'])

    if self.transform:
      image = self.transform(image)

    target = self.vocab.encode(self.data[idx]['text'])
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
    

class ClusterRandomSampler(Sampler):

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster, cluster_image in self.data_source.cluster_image.items():
            if self.shuffle:
                random.shuffle(cluster_image)

            batches = [
                cluster_image[i : i + self.batch_size]
                for i in range(0, len(cluster_image), self.batch_size)
            ]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)
        self.num_batch = len(batch_lists)
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)