import numpy as np
import torch
import lightning as L
from sklearn.metrics import f1_score, matthews_corrcoef

from scream.models.mlp import LinearModel


class LitLinearModel(L.LightningModule):
    def __init__(self, lr, input_dim, EPOCHS, steps_per_epoch, pos_weight, num_layers=3, hidden_units=256, dropout=0.0):
        super().__init__()
        self.model = LinearModel(input_dim, num_layers=num_layers, hidden_units=hidden_units, dropout=dropout)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lr = lr
        self.save_hyperparameters()
        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels = [], []
        self.val_logits, self.val_labels, self.val_true_labels = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch

    def shared_step(self, batch, stage: str):
        x, y, pm_ra = batch
        y_pred = self.model(x).squeeze()

        y_cwola = y[:, 0]
        y_true = y[:, 1]

        loss = self.criterion(y_pred, y_cwola)

        return loss, y_pred.detach().cpu(), y_cwola.detach().cpu(), y_true.detach().cpu()

    def training_step(self, batch, batch_idx):
        loss, train_pred, train_true, _ = self.shared_step(batch, stage='train')
        if loss is not None:
            self.log("train loss", loss, on_epoch=True, prog_bar=True)
            self.train_logits.append(train_pred)
            self.train_labels.append(train_true)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, val_pred, val_true, val_actual = self.shared_step(batch, stage='validation')
        if loss is not None:
            self.log("validation loss", loss, on_epoch=True)
            self.val_logits.append(val_pred)
            self.val_labels.append(val_true)
            self.val_true_labels.append(val_actual)

    def test_step(self, batch, batch_idx):
        loss, test_pred, test_true, _ = self.shared_step(batch, stage='test')
        if loss is not None:
            self.log("test loss", loss, on_epoch=True)
            print(test_pred.shape)
            print(test_true.shape)
            self.test_logits.append(test_pred)
            self.test_labels.append(test_true)

    def on_train_epoch_end(self):
        if len(self.train_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.train_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.train_labels).numpy()

        train_f1 = f1_score(y_true, preds)
        train_mcc = matthews_corrcoef(y_true, preds)

        self.log("train f1 score", train_f1)
        self.log("train MCC score", train_mcc)

        self.train_logits.clear()
        self.train_labels.clear()

    def on_validation_epoch_start(self):
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_true_labels.clear()

    def on_validation_epoch_end(self):
        if len(self.val_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.val_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        preds_80 = (probs >= 0.8).astype(int)
        y_true = torch.cat(self.val_labels).numpy()

        val_f1 = f1_score(y_true, preds)
        val_mcc = matthews_corrcoef(y_true, preds)

        self.log("validation f1 score", val_f1)
        self.log("validation MCC score", val_mcc)

        y_actual = torch.cat(self.val_true_labels).numpy()

        val_f1 = f1_score(y_actual, preds)
        val_mcc = matthews_corrcoef(y_actual, preds)

        val_f1_80 = f1_score(y_actual, preds_80)
        val_mcc_80 = matthews_corrcoef(y_actual, preds_80)

        self.log("True validation f1 score", val_f1)
        self.log("True validation MCC score", val_mcc)

        self.log("True validation f1 score (0.8 thresh)", val_f1_80)
        self.log("True validation MCC score (0.8 thresh)", val_mcc_80)

    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()

    def on_test_epoch_end(self):
        if len(self.test_logits) == 0:
            return
        probs = torch.sigmoid(torch.cat(self.test_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.test_labels).numpy()

        test_f1 = f1_score(y_true, preds)
        test_mcc = matthews_corrcoef(y_true, preds)

        self.log("test f1 score", test_f1)
        self.log("test MCC score", test_mcc)

        self.test_logits.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.EPOCHS,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=1e3
            ),
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
