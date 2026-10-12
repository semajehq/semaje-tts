"""Piper's training CLI with semaje's checkpoint callbacks.

Identical to `python -m piper.train` except for the default callbacks: Piper
always adds a val_mos ModelCheckpoint, which crashes at the first epoch end when
the UTMOS predictor cannot load, and Lightning appends config-file callbacks to
those defaults rather than replacing them. So the defaults are set here.
"""
import logging
import os

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel


def callbacks(use_mos: bool) -> list:
    out = [ModelCheckpoint(monitor="val_mel", mode="min", save_top_k=2, save_last=True,
                           filename="epoch={epoch}-val_mel={val_mel:.4f}", auto_insert_metric_name=False)]
    if use_mos:
        out.append(ModelCheckpoint(monitor="val_mos", mode="max", save_top_k=2, save_last=False,
                                   filename="epoch={epoch}-val_mos={val_mos:.4f}", auto_insert_metric_name=False))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_mos = os.environ.get("SEMAJE_TRAIN_MOS", "1") == "1"
    VitsLightningCLI(VitsModel, VitsDataModule,
                     trainer_defaults={"max_epochs": -1, "callbacks": callbacks(use_mos)})


if __name__ == "__main__":
    main()
