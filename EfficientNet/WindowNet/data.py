from typing import Callable, Optional, List, Tuple
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import pydicom

import json
import pandas as pd
from pathlib import Path

class MyDatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.label_column = "LessionTypeName"

    def load(self):
        with open(self.data_dir / r"C:\Users\Agah\Desktop\dcm_train_test\png_MR.json", "r") as f:
            label_entries = json.load(f)

        data = []
        for entry in label_entries:
            image_id = entry["ImageId"].replace(".png", ".dcm")
            label = entry[self.label_column]

            # split'e göre klasörleri tara
            for split in ["train", "val", "test"]:
                dcm_path = self.data_dir / split / image_id
                if dcm_path.exists():
                    data.append({
                        "Path": str(dcm_path),
                        "split": split,
                        "label": label
                    })
                    break

        full_df = pd.DataFrame(data)
        self.df_train = full_df[full_df["split"] == "train"]
        self.df_val = full_df[full_df["split"] == "val"]
        self.df_test = full_df[full_df["split"] == "test"]

        self.label_mapping = {name: i for i, name in enumerate(full_df["label"].unique())}
        for df in [self.df_train, self.df_val, self.df_test]:
            df["label_idx"] = df["label"].map(self.label_mapping)

        self.labels = ["label_idx"]




class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, load_image: Callable, transforms: Optional[Callable] = None):
        self.df = df
        self.load_image = load_image
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.load_image(row.Path)
        label = row.label_idx  # int

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()

        return img, label, index


def normalize_div_max(x):
    return x.div(x.max())

class Transforms():

    def __init__(self, labels, size, no_transforms, contrast=False, **kwargs):
        self.labels = labels
        self.size = (size, size)
        self.contrast = contrast
        self.transform = self._transform()
        self.no_transforms = no_transforms

    def _transform(self):
        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        pipeline = []
        augmentations = []
        augmentations.append(T.Normalize(mean, std))
        augmentations.append(T.Lambda(normalize_div_max))
        if self.contrast:
            augmentations.append(T.ColorJitter(brightness=0.5, contrast=0.5))
        pipeline.extend(augmentations)
        self.augmentations = T.Compose(augmentations)
        return T.Compose(pipeline)

    def __call__(self, input, target):
        if self.no_transforms:
            return input[0][None, :], self.target_transform(target)
        return self.transform(input), self.target_transform(target)

    def target_transform(self, target):
        return torch.tensor(target, dtype=torch.float32)


class DataModule(pl.LightningDataModule):

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--size", type=int, default=224)
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--pin_memory", default=True, action="store_true")
        parser.add_argument("--eight_bit", default=False, action="store_true")
        parser.add_argument("--center", default=None, type=int)
        parser.add_argument("--width", default=None, type=int)
        parser.add_argument("--samples", default=None, type=int)
        parser.add_argument("--no_transforms", default=False, action="store_true")
        parser.add_argument("--contrast_transform", default=False, action="store_true")
        return parent_parser

    @classmethod
    def from_argparse_args(cls, *args, **kwargs):
        """Pass the ArgParser's args to the constructor."""
        return pl.utilities.argparse.from_argparse_args(cls, *args, **kwargs)

    def __init__(
            self,
            batch_size,
            num_workers,
            pin_memory,
            size,
            eight_bit=False,
            center=None,
            width=None,
            cache=True,
            samples=None,
            no_transforms=False,
            contrast_transform=False,
            **kwargs
    ):
        super().__init__()
        self.batch_size = 16
        self.num_workers = 0
        self.pin_memory = True
        self.size = 224
        self.center = center
        self.width = width
        self.eight_bit = eight_bit
        self.cache = cache
        self.samples = samples
        self.no_transforms = no_transforms
        self.contrast_transform = contrast_transform

    def load_image(self, path):
        data = pydicom.read_file(path).pixel_array.astype("float32")

        # Normalize 0-1 aralığına
        data -= data.min()
        data /= data.max() + 1e-5

        tensor = TF.resize(TF.to_tensor(data), (self.size, self.size))
        tensor = tensor.repeat(3, 1, 1)

        return tensor

    def setup(self, stage=None):
        self.data = MyDatasetLoader(data_dir=r"C:\Users\Agah\Desktop\dcm_train_test")
        self.data.load()

    def dataloader(self, df, stage):
        self.transforms = Transforms(labels=self.data.labels,
                                     size=self.size,
                                     no_transforms=self.no_transforms,
                                     contrast=self.contrast_transform)
        setattr(self, stage + "_transforms", self.transforms)
        dataset = Dataset(df, self.load_image, self.transforms)
        setattr(self, stage + "_dataset", dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=stage == "train",
        )

    def train_dataloader(self):
        return self.dataloader(self.data.df_train if self.samples is None else self.data.df_train.iloc[:self.samples], "train")

    def val_dataloader(self):
        return self.dataloader(self.data.df_val, "val")

    def test_dataloader(self):
        return self.dataloader(self.data.df_test, "test")
