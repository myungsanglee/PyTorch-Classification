import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy

from utils.module_select import get_optimizer, get_scheduler
from models.loss.focal_loss import FocalLoss


class Classifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.top_1 = Accuracy(top_k=1)
        self.top_5 = Accuracy(top_k=5)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.ce_loss(y_pred, y)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.ce_loss(y_pred, y)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_top1', self.top_1(y_pred, y), logger=True, on_epoch=True, on_step=False)
        self.log('val_top5', self.top_5(y_pred, y), logger=True, on_epoch=True, on_step=False)
    
    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        try:
            scheduler = get_scheduler(
                cfg['scheduler'],
                optim,
                **cfg['scheduler_options']
            )
    
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            } 
        
        except KeyError:
            return optim
