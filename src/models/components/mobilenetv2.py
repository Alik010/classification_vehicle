import torch
from torch import nn
from torchvision.models import mobilenet_v2


class MobileNetV2(nn.Module):

    def __init__(self, out_features: int = 1) -> None:
        super().__init__()

        self.model = mobilenet_v2(pretrained=True)

        # Получение размера выходного признака последнего сверточного слоя
        num_ftrs = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features=num_ftrs, out_features=out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    _ = MobileNetV2()
