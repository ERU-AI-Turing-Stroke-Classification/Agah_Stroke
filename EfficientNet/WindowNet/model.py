from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torchvision.models import densenet121
from torchvision import transforms as T
from sklearn.metrics import f1_score


class ConvWindow(nn.Module):
    """Windowing as 1x1 convolution with clamping.
    Adapted from H. Lee et al., ‘Practical Window Setting Optimization for Medical Image Deep Learning’ (2018)
    """
    def __init__(self, widths=[4096, 4096, 4096], levels=[4096/2, 4096/2, 4096/2], upper=255, learn=True):
        super().__init__()
        assert len(widths) == len(levels), "The same number of widths and levels must be provided!"
        self.add_1x1 = len(widths) > 3 # pre-trained model expects three input channels
        self.learn = learn
        self.conv = nn.Conv2d(1, len(widths), kernel_size=(1, 1), stride=1)
        self.conv2 = nn.Conv2d(len(widths), 3, kernel_size=(1, 1), stride=1)
        if self.learn:
            for i, (width, level) in enumerate(zip(widths, levels)):
                self.conv.weight[i].data.fill_(255/width)
                self.conv.bias[i].data.fill_(-255/width * (level - width / 2))
        self.upper = upper

    def forward(self, x):
        out = self.conv(x)
        if self.learn:
            out = torch.clamp(out, min=0, max=self.upper)
        if self.add_1x1:
            out = self.conv2(out)
        return out

    def get_width_and_level(self):
        WW = (self.upper / self.conv.weight.detach()).flatten()
        WL = -(self.conv.bias.detach() / self.conv.weight.detach().flatten()) + WW / 2
        return WW, WL

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = self._transform()

    def _transform(self):
        # ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        pipeline = []
        augmentations = []
        augmentations.append(T.Normalize(mean, std))
        augmentations.append(T.Lambda(lambda x: x.div(x.max())))
        pipeline.extend(augmentations)
        self.augmentations = T.Compose(augmentations)
        return T.Compose(pipeline)

    def forward(self, x):
        return self.transform(x)


class Model(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--optim", type=str, default="adam")
        parser.add_argument("--scheduler", type=str, default=None)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--pretrained", default=False, action="store_true")
        parser.add_argument("--dropout", default=0, type=float)
        parser.add_argument("--classes", default=14, type=int)
        parser.add_argument("--state_dict_id", default=None)
        parser.add_argument("--state_dict_epoch", default=None)
        parser.add_argument("--learn_window", default=False, action="store_true")
        parser.add_argument("--freeze_classifier", default=False, action="store_true")
        parser.add_argument("--levels", type=str, default=None)
        parser.add_argument("--widths", type=str, default=None)
        parser.add_argument("--freeze_window", default=False, action="store_true")
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Pass the ArgParser's args to the constructor."""
        result = pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)
        # HACK: Log all Hyperparameters
        params = vars(args)
        params.update(**kwargs)
        result._hparams_initial = params
        return result

    def __init__(
            self,
            pretrained=True,
            learning_rate=1e-3,
            weight_decay=None,
            optim="adam",
            scheduler=None,
            patience=10,
            dropout=0,
            classes=14,
            state_dict_id=None,
            state_dict_epoch=None,
            learn_window=False,
            widths=None,
            levels=None,
            freeze_classifier=False,
            freeze_window=False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.state_dict_id = state_dict_id
        self.state_dict_epoch = state_dict_epoch
        self.freeze_classifier = freeze_classifier
        self.loss = torch.nn.CrossEntropyLoss()
        self.learn_window = learn_window
        self.freeze_window = freeze_window
        if self.learn_window:
            self.widths = [int(i) for i in widths.split(",")]
            self.levels = [int(i) for i in levels.split(",")]
        self._load_model()
        self.best_val = 0;
        self.best_epoch = 0;

    def _load_model(self):
        model = densenet121(pretrained=self.hparams.pretrained) #Buradan modeli değiştirebilirsiniz
        kernel_count = model.classifier.in_features
        model.classifier = torch.nn.Linear(kernel_count, 3)  # 3 sınıflı çıktı
        self.model = model

        if self.freeze_classifier:
            for param in self.model.parameters():
                param.requires_grad = False

        # Window işlemleri varsa
        if self.learn_window:
            self.window = ConvWindow(self.widths, self.levels, learn=not self.freeze_window)
            self.normalize = Normalize()
            WW, WL = self.window.get_width_and_level()
            print("Window widths: ", WW)
            print("Window levels: ", WL)
            for channel in range(WW.shape[0]):
                self.log(f"window/width/{channel}", WW[channel], prog_bar=True)
                self.log(f"window/level/{channel}", WL[channel], prog_bar=True)

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, images):

        if images.dim() == 3:
            images = images.unsqueeze(1)

        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        if self.learn_window:
            window = self.window(images)
            normalized = self.normalize(window)

            if normalized.dim() == 3:
                normalized = normalized.unsqueeze(1)

            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            y_hat = self.model(normalized)
        else:
            y_hat = self.model(images)
        return y_hat

    def _step(self, batch, batch_idx):
        images, y, indices = batch

        if images.dim() == 3:
            images = images.unsqueeze(1)

        y = y.view(-1)
        y_pred = self(images)


        loss = self.loss(y_pred, y)
        probs = torch.softmax(y_pred, dim=1).detach()
        logits = y_pred.detach()

        return {
            "loss": loss,
            "probs": probs,
            "labels": y,
            "logits": logits,
            "indices": indices,
        }

    def training_step_end(self, outputs):
        self.log("loss/train", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return outputs

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        if not hasattr(self, "train_outputs"):
            self.train_outputs = []
        self.train_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        probs = torch.vstack([o["probs"] for o in self.train_outputs]).cpu()
        labels = torch.cat([o["labels"] for o in self.train_outputs], dim=0).cpu()

        preds = torch.argmax(probs, dim=1)

        f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
        self.log("F1/train", f1, prog_bar=True, on_epoch=True)

        self.train_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)

        if not hasattr(self, "val_outputs"):
            self.val_outputs = []

        self.val_outputs.append({
            "probs": output["probs"].detach().cpu(),
            "targets": output["labels"].detach().cpu(),
        })

        self.log("val_loss", output["loss"])

        return output["loss"]

    def on_validation_epoch_end(self):
        if hasattr(self, "val_outputs") and len(self.val_outputs) > 0:
            probs = torch.cat([o["probs"] for o in self.val_outputs], dim=0)
            targets = torch.cat([o["targets"] for o in self.val_outputs], dim=0)

            preds = torch.argmax(probs, dim=1)
            f1 = f1_score(targets.numpy(), preds.numpy(), average="weighted")
            self.log("F1/val", f1, prog_bar=True, on_epoch=True)

            self.val_outputs = []  # Temizle

    def validation_step_end(self, outputs):
        self.log("loss/val", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return outputs

    def _epoch_end(self, outputs, stage):
        aucs = []

        probs = torch.vstack([o["probs"] for o in outputs]).cpu().numpy()
        labels = torch.vstack([o["labels"] for o in outputs]).int().cpu().numpy()
        for index, label in enumerate(self.trainer.datamodule.data.labels):
            try:
                aucs.append(roc_auc_score(labels[:,index], probs[:,index]))
                self.log(f"{label}/{stage}", aucs[-1])
            except ValueError as e:
                print(e)
                print(f"Could not calculate the AUC for {label} with {labels[:,index].sum().item()}")
                print(f"Stage {stage} with {len(labels)} samples.")
        if stage != "train":
            indices = torch.hstack([o["indices"] for o in outputs])
            logits = torch.vstack([o["logits"] for o in outputs])
            if len(aucs) > 0:
                mean_auc = sum(aucs)/len(aucs)
                self.log(f"Mean_AUC/{stage}", mean_auc, on_epoch=True, prog_bar=True)
                if mean_auc > self.best_val:
                    self.best_val = mean_auc
                    self.best_epoch = self.current_epoch
            self.log_prediction(stage=stage,
                                predictions=probs,
                                labels=labels,
                                indices=indices.cpu(),
                                logits=logits.cpu())



    def log_prediction(self, stage, filename=None, **kwargs):
        if hasattr(self.logger, "log_dir"):
            checkpoint = f"epoch={self.current_epoch}-step={self.global_step}"
            checkpoint_dir = Path(self.logger.log_dir) / "checkpoints" / checkpoint
            Path.mkdir(checkpoint_dir, parents=True, exist_ok=True)
            filename = checkpoint_dir / (f"{stage}-predictions.pt"
                                        if filename is None else filename)
            if not filename.exists():
                print(f"Saving predictions in {filename}")
                torch.save(kwargs, filename)


    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def test_step_end(self, outputs):
        self.log("loss/test", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return outputs

    def test_epoch_end(self, outputs):
        probs = torch.vstack([o["probs"] for o in outputs]).cpu()
        labels = torch.cat([o["labels"] for o in outputs], dim=0).cpu()

        preds = torch.argmax(probs, dim=1)
        f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
        self.log("F1/test", f1, prog_bar=True, on_epoch=True)

        self._epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.hparams.optim == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(),
                                        lr=self.hparams.learning_rate)

        if self.hparams.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", verbose=True, patience=self.hparams.patience)
            return ({"optimizer": optimizer,
                     "lr_scheduler": {"scheduler": self.scheduler,
                                      "monitor": "loss/val_epoch"}})
        return optimizer
