import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms.v2 as vT
from torchvision.io.image import decode_image, ImageReadMode


class ImageNetDataset(Dataset):
    def __init__(self, data_dir, image_paths_pkl, desired_size):
        self.root_dir = data_dir
        with open(image_paths_pkl, 'rb') as f:
            self.image_paths = pickle.load(f)
        self.transforms = vT.Compose(
            [
                vT.Resize(desired_size),
                vT.CenterCrop(desired_size),
                vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_paths[idx])
        image = decode_image(path, mode=ImageReadMode.RGB)
        image = self.transforms(image / 255.0)
        return image
