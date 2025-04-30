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

class Classification_ImageDataset(Dataset):
    def __init__(self, root_dir, image_paths_pkl=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # For Tiny-ImageNet directory structure (root/class/images/*.JPEG)
        for class_dir in self.classes:
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                images_dir = os.path.join(class_path, 'images')
                if os.path.isdir(images_dir):
                    class_images = glob(os.path.join(images_dir, '*.JPEG')) + \
                                   glob(os.path.join(images_dir, '*.jpg')) + \
                                   glob(os.path.join(images_dir, '*.png'))
                    self.image_paths.extend(class_images)
                    self.labels.extend([self.class_to_idx[class_dir]] * len(class_images))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = decode_image(path, mode=ImageReadMode.RGB)
        label = self.labels[idx]
        return image, label