import numpy as np
import torch
import lightning as L
from sklearn.metrics import f1_score, matthews_corrcoef

from scream.models.mlp import LinearModel
from scream.losses.mc_marginal import mc_marginal_bce_loss


class EM_LitLinearModel(L.LightningModule):
    # This model will implement EM with MC marginal BCE loss
    # To do so I will need to sample from the error distributions of the features during training and evaluation (before passing to the model)
    def __init__(self, lr, input_dim, EPOCHS, steps_per_epoch, pos_weight, num_layers=3,
                 hidden_units=256, dropout=0.0, num_mc_samples=10, pct_start=0.3, weight_decay=0.0, layer_norm=False,
                 activation='relu', residual=False, anneal_noise=False, noise_anneal_type='linear_decay', noise_anneal_dict=None):
        super().__init__()
        self.model = LinearModel(input_dim, num_layers=num_layers, hidden_units=hidden_units, dropout=dropout,
                                 layer_norm=layer_norm, activation=activation, residual=residual)
        self.pos_weight = pos_weight
        self.lr = lr
        self.num_mc_samples = num_mc_samples
        self.pct_start = pct_start
        self.weight_decay = weight_decay

        self.anneal_noise = anneal_noise
        self.noise_anneal_type = noise_anneal_type
        self.noise_anneal_dict = noise_anneal_dict
        self.save_hyperparameters()

        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels = [], []
        self.val_logits, self.val_labels, self.val_true_labels = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch

    def shared_step(self, batch, stage: str):
        x, y, errors, _ = batch
        # x shape: (B, D)
        # errors shape: (B, D)

        # Sample from the error distributions, num_mc_samples times
        B, D = x.shape
        noise_factor = self.noise_scale_factor()

        err_mask = torch.ones_like(errors, dtype=torch.bool)
        err_mask[:, 3] = False
        err_mask[:, 4] = False
        errors = torch.where(err_mask, noise_factor * errors, errors)

        x_samples = x.unsqueeze(1) + torch.randn(B, self.num_mc_samples, D).to(x.device) * errors.unsqueeze(1)

        assert x_samples.shape == (B, self.num_mc_samples, D), f"Expected shape (B, {self.num_mc_samples}, D), got {x_samples.shape}"

        if torch.isnan(x_samples).any() or torch.isinf(x_samples).any():
            raise RuntimeError("x_samples contains NaN/Inf")

        y_pred = self.model(x_samples).squeeze(-1)
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise RuntimeError("y_pred contains NaN/Inf")
        assert y_pred.shape == (B, self.num_mc_samples), f"Expected shape (B, {self.num_mc_samples}), got {y_pred.shape}"

        y_pred = y_pred.permute(1, 0)  # Shape: (num_mc_samples, B)

        y_cwola = y[:, 0]
        y_true = y[:, 1]

        loss = mc_marginal_bce_loss(y_pred, y_cwola, self.pos_weight)

        probs_mc = torch.sigmoid(y_pred)  # (N_mc, B)
        p_marginal = probs_mc.mean(dim=0)  # (B,)

        return loss, p_marginal.detach().cpu(), y_cwola.detach().cpu(), y_true.detach().cpu()

    def training_step(self, batch, batch_idx):
        loss, train_pred, train_true, _ = self.shared_step(batch, stage='train')
        if loss is not None:
            self.log("train loss", loss, on_epoch=True)
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
        probs = torch.cat(self.train_logits).numpy()
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
        probs = torch.cat(self.val_logits).numpy()
        assert (probs.min() >= 0 and probs.max() <= 1), f'Probabilities are out of [0,1] range'
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

        self.log("True validation f1 score (0.8 thresh)", val_f1_80, prog_bar=True)
        self.log("True validation MCC score (0.8 thresh)", val_mcc_80)

    def on_test_epoch_start(self):
        self.test_logits.clear()
        self.test_labels.clear()

    def on_test_epoch_end(self):
        if len(self.test_logits) == 0:
            return
        probs = torch.cat(self.test_logits).numpy()
        preds = (probs >= 0.5).astype(int)
        y_true = torch.cat(self.test_labels).numpy()

        test_f1 = f1_score(y_true, preds)
        test_mcc = matthews_corrcoef(y_true, preds)

        self.log("test f1 score", test_f1)
        self.log("test MCC score", test_mcc)

        self.test_logits.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        if self.weight_decay > 0.0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.EPOCHS,
                pct_start=self.pct_start,
                div_factor=10.0,
                final_div_factor=1e3
            ),
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_after_backward(self):
        """Check for NaN/Inf or exploding/vanishing gradients."""
        grad_norm = 0.0
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            g = p.grad

            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"[Gradient NaN/Inf detected] in {name}")
                self.log("debug/grad_nan_inf", 1)
                return

            grad_norm += g.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5
        self.log("debug/grad_norm", grad_norm, on_step=True, on_epoch=False)

        if grad_norm < 1e-6:
            print(f"[Vanishing Gradients] Global grad norm = {grad_norm:.2e}")
            self.log("debug/vanishing_grad", grad_norm)

    def on_before_optimizer_step(self, optimizer):
        """Check parameter values before the optimizer updates them."""
        for name, p in self.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[Parameter NaN/Inf detected] in {name}")
                self.log("debug/param_nan_inf", 1)
                return

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        total = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))
        self.log("debug/grad_norm_post_clip", total, on_step=True)
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Check model outputs for numerical issues."""
        loss = outputs['loss']
        if torch.isnan(loss) or torch.isinf(loss):
            print("[Loss NaN/Inf detected]")
            self.log("debug/loss_nan_inf", 1)

    def noise_scale_factor(self):
        """Return a scalar in [noise_anneal_min, 1] depending on training progress."""
        if not self.anneal_noise or not self.training:
            return 1.0

        total_steps = self.steps_per_epoch * self.EPOCHS
        t = min(1.0, self.global_step / total_steps)

        noise_anneal_min = self.noise_anneal_dict['noise_anneal_min']

        if self.noise_anneal_type == "linear_decay":
            return 1.0 - (1.0 - noise_anneal_min) * t

        elif self.noise_anneal_type == "cosine":
            f_max = self.noise_anneal_dict['f_max']
            t_peak = self.noise_anneal_dict['t_peak']

            if t < t_peak:
                return 1 + 0.5 * (f_max - 1) * (1 - np.cos(np.pi * t / t_peak))
            else:
                return noise_anneal_min + 0.5 * (f_max - noise_anneal_min) * (
                    1 + np.cos(np.pi * (t - t_peak) / (1 - t_peak))
                )

        elif self.noise_anneal_type == "exp":
            return noise_anneal_min + (1 - noise_anneal_min) * np.exp(-5 * t)

        return 1.0
