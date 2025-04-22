from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
from modules.detection.utils import generate_quad, resize_and_pad, transform, generate_score_mask
   
from configs.detection_config import target_size

# Định nghĩa lớp Dataset tùy chỉnh
class MSRADataset(Dataset):
  def __init__(self, data, dset): 
    super(MSRADataset, self).__init__()
    self.data = data[dset]
    self.dset = dset 

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    # image
    image_name = self.data[idx]['img_name']
    image_path = f"datasets/MSRA-TD500/{self.dset}/{image_name}"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, raito = resize_and_pad(image)
    # boxes
    polygons=[]
    for box in self.data[idx]['boxes']:
      polygons.append((np.array(box['polylines'])*raito).astype(int))


    score_maps_gt, geo_maps_gt, training_masks = generate_quad((image.shape[0], image.shape[1]), polygons)

    image = transform(image).astype(np.float32)
    geo_maps_gt /= target_size[0]

    return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(score_maps_gt), torch.from_numpy(geo_maps_gt), torch.from_numpy(training_masks)
  
class MSRATD500Dataset(Dataset):
  def __init__(self, data, dset): 
    super(MSRATD500Dataset, self).__init__()
    self.data = data[dset]
    self.dset = dset 

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    # image
    image_name = self.data[idx]['img_name']
    image_path = f"datasets/MSRA-TD500/{self.dset}/{image_name}"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, raito = resize_and_pad(image)

    # boxes
    polygons=[]
    for box in self.data[idx]['boxes']:
      polygons.append((np.array(box['polylines'])*raito).astype(int))

    score_maps_gt, training_masks = generate_score_mask((image.shape[0], image.shape[1]), polygons)

    image = transform(image)

    return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1),  \
            torch.from_numpy(score_maps_gt.astype(np.float32)),          \
            torch.from_numpy(training_masks.astype(np.float32))