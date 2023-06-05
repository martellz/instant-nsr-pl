import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

from vcpy.viewcamera import *
from vcpy.linearfitting import sphere_fit

import datasets
from models.ray_utils import get_ray_directions_vc, get_c2w


class OppenDatasetBase():
  def setup(self, config, split):
    print(config)
    self.config = config
    self.rank = _get_rank()
    self.split = split

    data_root = Path(self.config.scene)
    img_dir = data_root / 'cropped'
    vcs = load_view_camera_data(data_root/ 'cams_view_frustum_cropped.bson')

    self.w = None
    self.h = None

    self.directions = []
    self.all_c2w = []
    self.all_images = []
    # self.all_fg_masks = []
    self.ws = []
    self.hs = []

    for cam in vcs.cameras:
      self.directions.append(get_ray_directions_vc(cam, self.config.use_pixel_centers))
      self.all_c2w.append(get_c2w(cam)[:3, :4])

      img = Image.open(img_dir / cam.name)
      # img = img.resize(self.config.img_wh, Image.BICUBIC)
      img = TF.to_tensor(img).permute(1, 2, 0) # (3, h, w) => (h, w, 3)
      self.all_images.append(img)

      self.ws.append(cam.width)
      self.hs.append(cam.height)

      if self.split != 'train':
        break

    cam_poss = []
    for c2w in self.all_c2w:
      cam_pos = c2w[:3, 3]
      cam_poss.append(cam_pos.tolist())
    # print(cam_poss)
    r, x, y, z = sphere_fit(cam_poss)
    r = r[0]
    center = np.array([x[0], y[0], z[0]])

    for c2w in self.all_c2w:
      cam_pos = c2w[:3, 3]
      cam_pos = (cam_pos - center) * 1.0 / self.config.obj_size
      c2w[:3, 3] = cam_pos

    self.all_c2w = [torch.from_numpy(c2w) for c2w in self.all_c2w]


    self.all_c2w = torch.stack(self.all_c2w, dim=0).float().to(self.rank)


class OppenDataset(Dataset, OppenDatasetBase):
  def __init__(self, config, split):
    self.setup(config, split)

  def __len__(self):
    return len(self.all_images)

  def __getitem__(self, index):
    return {
          'index': index
      }


class OppenIterableDataset(IterableDataset, OppenDatasetBase):
  def __init__(self, config, split):
    self.setup(config, split)

  def __iter__(self):
    while True:
      yield {}



@datasets.register('oppen')
class OppenDataModule(pl.LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def setup(self, stage=None):
    if stage in [None, 'fit']:
      self.train_dataset = OppenIterableDataset(self.config, self.config.train_split)
    # if stage in [None, 'fit', 'validate']:
    #   self.val_dataset = OppenDataset(self.config, self.config.val_split)
    # if stage in [None, 'test']:
    #   self.test_dataset = OppenDataset(self.config, self.config.test_split)
    # if stage in [None, 'predict']:
    #   self.predict_dataset = OppenDataset(self.config, self.config.train_split)

  def prepare_data(self):
    pass

  def general_loader(self, dataset, batch_size):
    sampler = None
    return DataLoader(
        dataset,
        num_workers=os.cpu_count(),
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler
    )


  def train_dataloader(self):
    return self.general_loader(self.train_dataset, batch_size=1)

  def val_dataloader(self):
    # return self.general_loader(self.val_dataset, batch_size=1)
    return None

  def test_dataloader(self):
    # return self.general_loader(self.test_dataset, batch_size=1)
    return None

  def predict_dataloader(self):
    # return self.general_loader(self.predict_dataset, batch_size=1)
    return None
