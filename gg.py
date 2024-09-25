from src.data.dataset import VehicleDataset
from src.models.vehicle_module import VehicleModule

labels = {'bus': 0,
          'car': 1,
          'minibus': 2,
          'minitruck': 3,
          'trailer': 4,
          'truck': 5,
          'truck_trailer': 6,
        }

data = VehicleDataset(path_image_data="dataset/images",
                      path_annotation_data="dataset/annotations/test.json",
                      size="(224, 224)", labels=labels, augmentation=None)

checkpoint_name = r"logs/train/runs/2024-09-12_07-22-19/checkpoints/epoch_065.ckpt"
DEVICE = 'cpu'

module = VehicleModule.load_from_checkpoint(checkpoint_name)
module.eval()
module.to(DEVICE)

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from typing import List, Optional, Tuple
import numpy as np

def denormalize(
        img: NDArray[float],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        max_value: int = 255,
) -> NDArray[int]:
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)['image'] * max_value
    return denorm_img.astype(np.uint8)

import os
from PIL import Image
from torch.utils.data import DataLoader
import torch

output_folder = "predicted_images"
misclassified_folder = "misclassified_images"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(misclassified_folder, exist_ok=True)

# DataLoader for test dataset
dataloader = DataLoader(data, batch_size=1, shuffle=False)

# Iterate through test dataset
for i, (image, label) in enumerate(dataloader):
    image = image.to(DEVICE)
    label = label.item()

    # Make prediction
    with torch.no_grad():
        prediction = module(image)
        predicted_class_idx = torch.argmax(prediction, dim=1).item()

    # Find predicted class label
    predicted_class = [key for key, value in labels.items() if value == predicted_class_idx][0]
    true_class = [key for key, value in labels.items() if value == label][0]

    # Get image filename
    img_name = f"image_{i}_pred_{predicted_class}_true_{true_class}.jpg"

    # Convert tensor to numpy array and denormalize
    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy array
    denorm_img = denormalize(image_np)  # Apply the denormalize function

    # Save the image in appropriate folder
    if predicted_class_idx == label:
        # Correct prediction: Save to predicted_images folder
        img = Image.fromarray(denorm_img)
        img.save(os.path.join(output_folder, img_name))
    else:
        # Incorrect prediction: Save to misclassified_images folder
        img = Image.fromarray(denorm_img)
        img.save(os.path.join(misclassified_folder, img_name))

print("Predictions and misclassifications saved!")

