from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI


from model import Model
from data import DataModule

#Kodu çalıştırmak için terminale şunu yazın (dosya yolunu değiştirerek): python main.py fit --config config.yaml

#Kodun orijinalinde olan yapı (lightning kütüphanesi artık bunu kullanmıyormuş o yüzden kullanamıyoruz).
"""
def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("Main")
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    return parent_parser


def run():
    parser = ArgumentParser()
    parser = add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = Model.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args, unknown = parser.parse_known_args()
    dict_args = vars(args)

    pl.seed_everything(dict_args["seed"], workers=True)
    datamodule = DataModule.from_argparse_args(args)
    model = Model.from_argparse_args(args)
    logger = TensorBoardLogger("experiments")
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        save_top_k=50,
        monitor="Mean_AUC/val",
        verbose=True,
        save_last=True,
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="loss/val_epoch",
        min_delta=0.00,
        patience=dict_args["early_stopping_patience"],
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=[checkpoint_callback,
                                        early_stop_callback]
    )
    trainer.fit(model, datamodule)
    model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, datamodule)

"""
if __name__ == "__main__":
    LightningCLI(model_class=Model, datamodule_class=DataModule)
