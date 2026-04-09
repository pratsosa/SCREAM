import numpy as np
import torch
import lightning as L
from sklearn.metrics import f1_score, matthews_corrcoef

from scream.models.mlp import LinearModel
from scream.losses.mc_marginal import mc_marginal_bce_loss
from scream.data.photometry import (
    flux_to_mag_gaia, mag_to_flux_gaia,
    flux_to_mag_ls, mag_to_flux_ls,
    extinction_gaia, extinction_ls,
    ZP_G, ZP_BP, ZP_RP,
)


class EM_LitLinearModel(L.LightningModule):
    # This model will implement EM with MC marginal BCE loss
    # To do so I will need to sample from the error distributions of the features during training and evaluation (before passing to the model)
    def __init__(self, lr, input_dim, EPOCHS, steps_per_epoch, pos_weight,
                 scaler_mean: np.ndarray, scaler_scale: np.ndarray,
                 n_extinction_iter: int = 10,
                 num_layers=3, hidden_units=256, dropout=0.0, num_mc_samples=10,  # hidden_units: int or list[int]
                 pct_start=0.3, weight_decay=0.0, layer_norm=False,
                 activation='relu', residual=False):
        super().__init__()
        self.model = LinearModel(input_dim, num_layers=num_layers, hidden_units=hidden_units, dropout=dropout,
                                 layer_norm=layer_norm, activation=activation, residual=residual)
        self.pos_weight = pos_weight
        self.lr = lr
        self.num_mc_samples = num_mc_samples
        self.pct_start = pct_start
        self.weight_decay = weight_decay
        self.n_extinction_iter = n_extinction_iter

        self.register_buffer('scaler_mean',
                             torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer('scaler_scale',
                             torch.tensor(scaler_scale, dtype=torch.float32))

        self.save_hyperparameters(ignore=['scaler_mean', 'scaler_scale'])

        self.logits = []
        self.labels = []
        self.train_logits, self.train_labels = [], []
        self.val_logits, self.val_labels, self.val_true_labels = [], [], []
        self.test_logits, self.test_labels = [], []
        self.EPOCHS = EPOCHS
        self.steps_per_epoch = steps_per_epoch

    def shared_step(self, batch, stage: str):
        x_raw, y, errors, id_plus_sample, *_ = batch
        # x_raw shape: (B, 10)  — phi1, phi2, pm_phi1, pm_phi2, G_mag, Bp_mag, Rp_mag, g_mag, r_mag, z_mag
        # errors shape: (B, 11) — phot_g_flux_err, phot_bp_flux_err, phot_rp_flux_err,
        #                          flux_err_g, flux_err_r, flux_err_z,
        #                          pmra_error, pmdec_error, ra_error, dec_error,
        #                          ebv (index 10)

        B = x_raw.shape[0]
        N_mc = self.num_mc_samples

        # --- Step 1: Unpack ---
        phi1, phi2, pm_phi1, pm_phi2 = x_raw[:, 0], x_raw[:, 1], x_raw[:, 2], x_raw[:, 3]
        G_mag, Bp_mag, Rp_mag        = x_raw[:, 4], x_raw[:, 5], x_raw[:, 6]
        g_mag, r_mag, z_mag          = x_raw[:, 7], x_raw[:, 8], x_raw[:, 9]

        phot_flux_errs              = errors[:, :6]      # (B, 6): G_err, Bp_err, Rp_err, g_err, r_err, z_err
        pmra_err, pmdec_err         = errors[:, 6], errors[:, 7]
        ra_err, dec_err             = errors[:, 8], errors[:, 9]   # degrees (converted from mas in GD1_data_prep.py)
        ebv                         = errors[:, 10]                # (B,)

        # --- Step 2: Convert photometric mags -> fluxes (B,) each ---
        flux_G  = mag_to_flux_gaia(G_mag,  ZP_G)
        flux_Bp = mag_to_flux_gaia(Bp_mag, ZP_BP)
        flux_Rp = mag_to_flux_gaia(Rp_mag, ZP_RP)
        flux_g  = mag_to_flux_ls(g_mag)
        flux_r  = mag_to_flux_ls(r_mag)
        flux_z  = mag_to_flux_ls(z_mag)

        # --- Step 3: Stack fluxes and errors (B, 6) ---
        fluxes = torch.stack([flux_G, flux_Bp, flux_Rp, flux_g, flux_r, flux_z], dim=1)

        # --- Step 4: Sample N_mc flux realisations (B, N_mc, 6) ---
        noise          = torch.randn(B, N_mc, 6, device=x_raw.device)
        fluxes_sampled = fluxes.unsqueeze(1) + noise * phot_flux_errs.unsqueeze(1)
        fluxes_sampled = fluxes_sampled.clamp(min=1e-10)

        # --- Step 5: Convert sampled fluxes back to magnitudes (B, N_mc) each ---
        G_s  = flux_to_mag_gaia(fluxes_sampled[:, :, 0], ZP_G)
        Bp_s = flux_to_mag_gaia(fluxes_sampled[:, :, 1], ZP_BP)
        Rp_s = flux_to_mag_gaia(fluxes_sampled[:, :, 2], ZP_RP)
        g_s  = flux_to_mag_ls(fluxes_sampled[:, :, 3])
        r_s  = flux_to_mag_ls(fluxes_sampled[:, :, 4])
        z_s  = flux_to_mag_ls(fluxes_sampled[:, :, 5])

        # --- Step 6: Apply extinction correction (B, N_mc) each ---
        ebv_e = ebv.unsqueeze(1).expand(B, N_mc)
        AG, ABp, ARp = extinction_gaia(G_s, Bp_s, Rp_s, ebv_e, n_iter=self.n_extinction_iter)
        Ag, Ar, Az   = extinction_ls(ebv_e)
        G0  = G_s  - AG;  Bp0 = Bp_s - ABp;  Rp0 = Rp_s - ARp
        g0  = g_s  - Ag;  r0  = r_s  - Ar;   z0  = z_s  - Az

        # --- Step 7: Compute colors (B, N_mc) ---
        BpRp0 = Bp0 - Rp0
        gr0   = g0  - r0
        rz0   = r0  - z0
        # r0 is used directly as rmag0 (extinction-corrected LS r-band magnitude)

        # --- Step 8: Sample astrometric errors (B, N_mc) each ---
        # pmra_error / pmdec_error used as proxies for pm_phi1 / pm_phi2 stream-frame errors
        # ra_error / dec_error already in degrees (converted from mas in GD1_data_prep.py)
        # TECH DEBT: proper treatment requires Jacobian of the stream-frame transform
        ast_noise = torch.randn(B, N_mc, 4, device=x_raw.device)
        phi1_s    = phi1.unsqueeze(1) + ast_noise[:, :, 0] * ra_err.unsqueeze(1)
        phi2_s    = phi2.unsqueeze(1) + ast_noise[:, :, 1] * dec_err.unsqueeze(1)
        pm_phi1_s = pm_phi1.unsqueeze(1) + ast_noise[:, :, 2] * pmra_err.unsqueeze(1)
        pm_phi2_s = pm_phi2.unsqueeze(1) + ast_noise[:, :, 3] * pmdec_err.unsqueeze(1)

        # --- Step 9: Stack final MLP input (B, N_mc, 9) ---
        x_samples = torch.stack([phi1_s, phi2_s, pm_phi1_s, pm_phi2_s,
                                  G0, BpRp0, r0, gr0, rz0], dim=-1)

        # --- Step 10: Scale using registered buffers (B, N_mc, 9) ---
        x_scaled = (x_samples - self.scaler_mean) / self.scaler_scale

        if torch.isnan(x_scaled).any() or torch.isinf(x_scaled).any():
            raise RuntimeError("x_scaled contains NaN/Inf")

        # --- Step 11: Forward pass ---
        y_pred = self.model(x_scaled).squeeze(-1)   # (B, N_mc)

        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise RuntimeError("y_pred contains NaN/Inf")
        assert y_pred.shape == (B, N_mc), f"Expected shape (B, {N_mc}), got {y_pred.shape}"

        y_pred = y_pred.permute(1, 0)  # Shape: (N_mc, B)

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
