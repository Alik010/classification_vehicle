from typing import Any, Dict, Tuple
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class VehicleModule(LightningModule):

    def __init__(
        self,
        num_classes: int,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        criterion: torch.nn
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = criterion

        self.train_accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.train_precision_metric = MulticlassPrecision(num_classes=num_classes)
        self.train_recall_metric = MulticlassRecall(num_classes=num_classes)
        self.train_f1_metric = MulticlassF1Score(num_classes=num_classes)

        self.val_accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision_metric = MulticlassPrecision(num_classes=num_classes)
        self.val_recall_metric = MulticlassRecall(num_classes=num_classes)
        self.val_f1_metric = MulticlassF1Score(num_classes=num_classes)

        self.test_accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision_metric = MulticlassPrecision(num_classes=num_classes)
        self.test_recall_metric = MulticlassRecall(num_classes=num_classes)
        self.test_f1_metric = MulticlassF1Score(num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation f1
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y.long())
        preds = F.softmax(preds, dim=1)

        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)


        self.train_accuracy_metric.update(preds, targets)
        self.train_precision_metric.update(preds, targets)
        self.train_recall_metric.update(preds, targets)
        self.train_f1_metric.update(preds, targets)

        self.log("loss/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        self.log("train/acc", self.train_accuracy_metric.compute(), prog_bar=True)
        self.log("train/P", self.train_precision_metric.compute(), prog_bar=True)
        self.log("train/R", self.train_recall_metric.compute(), prog_bar=True)
        self.log("train/F1", self.train_f1_metric.compute(), prog_bar=True)

        self.train_accuracy_metric.reset()
        self.train_precision_metric.reset()
        self.train_recall_metric.reset()
        self.train_f1_metric.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_accuracy_metric.update(preds, targets)
        self.val_precision_metric.update(preds, targets)
        self.val_recall_metric.update(preds, targets)
        self.val_f1_metric.update(preds, targets)

        self.log("loss/val", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        f1 = self.val_f1_metric.compute()  # get current val acc
        self.val_f1_best(f1)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True
        )

        self.log("val/acc", self.val_accuracy_metric.compute(), prog_bar=True)
        self.log("val/P", self.val_precision_metric.compute(), prog_bar=True)
        self.log("val/R", self.val_recall_metric.compute(), prog_bar=True)
        self.log("val/F1", self.val_f1_metric.compute(), prog_bar=True)

        self.val_accuracy_metric.reset()
        self.val_precision_metric.reset()
        self.val_recall_metric.reset()
        self.val_f1_metric.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_accuracy_metric.update(preds, targets)
        self.test_precision_metric.update(preds, targets)
        self.test_recall_metric.update(preds, targets)
        self.test_f1_metric.update(preds, targets)

        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/acc", self.test_accuracy_metric.compute(), prog_bar=True)
        self.log("test/P", self.test_precision_metric.compute(), prog_bar=True)
        self.log("test/R", self.test_recall_metric.compute(), prog_bar=True)
        self.log("test/F1", self.test_f1_metric.compute(), prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = VehicleModule(None, None, None, None)
