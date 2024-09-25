import torch
from torch import nn
import timm


class MobileNetV4(nn.Module):

    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()

        self.model = timm.create_model(model_name,
                                       pretrained=True)

        self.model.classifier = nn.Linear(in_features=1280,
                                          out_features=num_classes,
                                          bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    _ = MobileNetV4()
