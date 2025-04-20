import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms.v2 as vT
from torchvision.io.image import decode_image, ImageReadMode
from glob import glob


class ImageDataset(Dataset):
    def __init__(self, root_dir, image_paths_pkl=None):
        self.root_dir = root_dir
        if image_paths_pkl is None:
            self.image_paths = glob(os.path.join(root_dir, '*'))
        else:
            with open(image_paths_pkl, 'rb') as f:
                self.image_paths = pickle.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_paths[idx])
        image = decode_image(path, mode=ImageReadMode.RGB)
        return image
