"""Lightning CNN and the optimizer registry used by the comparison experiment."""
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from optimizers.custom import AdaBelief, Lion
import optimizers.optimizers as O

# name -> (factory(params, lr), learning-rate grid for the sweep)
OPTIMIZERS = {
    "SGD":       (lambda p, lr: torch.optim.SGD(p, lr=lr),                              [0.01, 0.03, 0.1]),
    "Momentum":  (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),               [0.003, 0.01, 0.03]),
    "Nesterov":  (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True), [0.003, 0.01, 0.03]),
    "AdaGrad":   (lambda p, lr: torch.optim.Adagrad(p, lr=lr),                         [0.01, 0.03, 0.1]),
    "AdaDelta":  (lambda p, lr: torch.optim.Adadelta(p, lr=lr),                        [0.3, 1.0, 3.0]),
    "RMSProp":   (lambda p, lr: torch.optim.RMSprop(p, lr=lr),                         [1e-4, 3e-4, 1e-3]),
    "Adam":      (lambda p, lr: torch.optim.Adam(p, lr=lr),                            [3e-4, 1e-3, 3e-3]),
    "AMSGrad":   (lambda p, lr: torch.optim.Adam(p, lr=lr, amsgrad=True),              [3e-4, 1e-3, 3e-3]),
    "AdamW":     (lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=1e-4),        [3e-4, 1e-3, 3e-3]),
    "Nadam":     (lambda p, lr: torch.optim.NAdam(p, lr=lr),                           [3e-4, 1e-3, 3e-3]),
    "RAdam":     (lambda p, lr: torch.optim.RAdam(p, lr=lr),                           [3e-4, 1e-3, 3e-3]),
    "AdaBelief": (lambda p, lr: AdaBelief(p, lr=lr),                           [3e-4, 1e-3, 3e-3]),
    "Lion":      (lambda p, lr: Lion(p, lr=lr, weight_decay=0.1),              [3e-5, 1e-4, 3e-4]),
}


MODEL_CONFIG = {
    "model": "FashionCNN-v1-pytorch-optimizers",
    "scheduler": "OneCycleLR",
    "optimizer_implementation": "torch.optim plus custom AdaBelief and Lion",
    "max_lr_factor": 10,
    "cycle_momentum": False,
    "adamw_weight_decay": 1e-4,
}


def model_config():
    return dict(MODEL_CONFIG)


def conv_block(c_in, c_out):
    """Two Conv-BN-ReLU layers followed by pooling and spatial dropout."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.1),
    )


class FashionCNN(pl.LightningModule):
    """Fashion-MNIST CNN trained with OneCycleLR."""

    def __init__(self, lr=1e-3, weight_decay=1e-4, num_classes=10,
                 max_epochs=30, optimizer_name=None):
        super().__init__()
        self.save_hyperparameters()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
        self.train_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.test_cm = torchmetrics.ConfusionMatrix("multiclass", num_classes=num_classes)
        self.iter_losses = []
        self.train_losses = []
        self.eval_losses = []
        self.eval_accs = []
        self._train_loss_sum = 0.0
        self._train_example_count = 0
        self._eval_loss_sum = 0.0
        self._eval_correct_count = 0
        self._eval_example_count = 0

    def forward(self, images):
        return self.classifier(self.features(images))

    def _step(self, batch):
        images, targets = batch
        logits = self(images)
        return F.cross_entropy(logits, targets), logits, targets

    def on_train_epoch_start(self):
        self._train_loss_sum = 0.0
        self._train_example_count = 0

    def training_step(self, batch, batch_idx):
        loss, logits, targets = self._step(batch)
        batch_size = targets.size(0)
        loss_value = loss.detach().item()
        self.iter_losses.append(loss_value)
        self._train_loss_sum += loss_value * batch_size
        self._train_example_count += batch_size
        self.train_acc.update(logits, targets)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self._train_example_count:
            self.train_losses.append(self._train_loss_sum / self._train_example_count)

    def on_validation_epoch_start(self):
        self._eval_loss_sum = 0.0
        self._eval_correct_count = 0
        self._eval_example_count = 0

    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._step(batch)
        batch_size = targets.size(0)
        self._eval_loss_sum += loss.detach().item() * batch_size
        self._eval_correct_count += (logits.argmax(dim=1) == targets).sum().item()
        self._eval_example_count += batch_size
        self.val_acc.update(logits, targets)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def on_validation_epoch_end(self):
        if self._eval_example_count and not self.trainer.sanity_checking:
            self.eval_losses.append(self._eval_loss_sum / self._eval_example_count)
            self.eval_accs.append(self._eval_correct_count / self._eval_example_count)

    def on_test_epoch_start(self):
        self.test_cm.reset()

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._step(batch)
        self.test_acc.update(logits, targets)
        self.test_cm.update(logits.argmax(1), targets)
        self.log("test/loss", loss, batch_size=targets.size(0))
        self.log("test/acc", self.test_acc)

    def predict_step(self, batch, batch_idx):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(images).softmax(dim=1)

    def configure_optimizers(self):
        if self.hparams.optimizer_name is None:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            factory, _ = OPTIMIZERS[self.hparams.optimizer_name]
            optimizer = factory(self.parameters(), self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr * 10,
            total_steps=self.trainer.estimated_stepping_batches,
            cycle_momentum=False,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


CNN = FashionCNN