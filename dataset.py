import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms.v2 as vT
from torchvision.io.image import decode_image, ImageReadMode
from glob import glob


class ImageDataset(Dataset):
    def __init__(self, root_dir, desired_size, image_paths_pkl=None):
        self.root_dir = root_dir
        if image_paths_pkl is None:
            self.image_paths = glob(os.path.join(root_dir, '*'))
        else:
            with open(image_paths_pkl, 'rb') as f:
                self.image_paths = pickle.load(f)

        self.transforms = vT.Compose(
            [
                vT.Resize(desired_size),
                vT.CenterCrop(desired_size),
                vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.desired_size = desired_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_paths[idx])
        image = decode_image(path, mode=ImageReadMode.RGB)
        image = self.transforms(image / 255.0)
        return image
