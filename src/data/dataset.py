from typing import Optional, Callable
import json
import torch
import albumentations as A
from torch.utils.data import Dataset
import numpy as np
import cv2
from src.data.components.augmentation import get_training_augmentation
class VehicleDataset(Dataset):

    def __init__(self,
                 path_annotation_data: str,
                 path_image_data: str,
                 labels: dict,
                 size: tuple,
                 augmentation: Optional[Callable] = None):

        self.path_annotation_data = path_annotation_data
        self.path_image_data = path_image_data
        self.labels = labels
        self.augmentation = augmentation
        self.size = tuple(map(int, size.strip('()').split(',')))

        height, width = self.size
        self.resize = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize()
        ])

        with open(self.path_annotation_data, 'r') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):

        image_name, label_str = self.data[idx]["file_name"], self.data[idx]["type"]
        label = self.label2num(label_str)
        image_full_path = f"{self.path_image_data}/{label_str}/{image_name}"
        image_bgr = cv2.imread(image_full_path)
        image_arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            sample = self.augmentation(image=image_arr)
            image_arr = sample['image']

        sample = self.resize(image=image_arr)
        image_arr = sample['image']

        image = torch.tensor(np.transpose(image_arr, (2, 0, 1)), dtype=torch.float32)

        return image, label

    def label2num(self, label):
        # label_zero = np.zeros(len(self.labels), dtype=int)
        target_label = self.labels[label]
        # label_zero[target_label] = 1
        label_convert = torch.tensor(target_label, dtype=torch.float)

        return label_convert




# import matplotlib.pyplot as plt
#
# from typing import List, Optional, Tuple
# from torch import Tensor
# from torchvision.utils import make_grid
#
# import albumentations as albu
# from albumentations.pytorch import ToTensorV2
# from numpy.typing import NDArray
#
#
# def denormalize(
#         img: NDArray[float],
#         mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
#         std: Tuple[float, ...] = (0.229, 0.224, 0.225),
#         max_value: int = 255,
# ) -> NDArray[int]:
#     denorm = albu.Normalize(
#         mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
#         std=[1.0 / st for st in std],
#         always_apply=True,
#         max_pixel_value=1.0,
#     )
#     denorm_img = denorm(image=img)['image'] * max_value
#     return denorm_img.astype(np.uint8)
#
#
# labels = {'bus': 0,
#          'car': 1,
#          'minibus': 2,
#          'minitruck': 3,
#          'trailer': 4,
#          'truck': 5,
#          'truck_trailer': 6,
#          'motorhome': 7}
#
# data = VehicleDataset(path_image_data="dataset/images",
#                       path_annotation_data="dataset/annotations/train.json",
#                       size="(128, 128)", labels=labels,augmentation=get_training_augmentation(r"C:\Users\aliko\PycharmProjects\cls_vehicle\configs\data\augmentation.yaml"))
#
# for i in range(len(data)):
#     image, label = data[i]
#     print(label)
#     image_np = image.squeeze(dim=0).permute(1, 2, 0).numpy()
#     image_np = denormalize(image_np)
#     plt.imshow((image_np).astype(np.uint8))
#     plt.show()
