# system, numpy
import os
import sys
import numpy as np
import math
from einops import rearrange, repeat
import einops
import opt_einsum
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# user defined
from src.optimizer import SAM

try:
    import snntorch as snn
    from snntorch import surrogate as snn_surrogate
except ImportError:  # optional until user runs: pip install snntorch
    snn = None
    snn_surrogate = None

torch.set_printoptions(threshold=10_000)


class FakeSNNConversion(nn.Module):
    """
    Minimal fake-SNN conversion for prototype-preserving ANN2SNN.

    Approximates low-step firing-rate quantization:
      x_clip = clamp(x, 0, threshold)
      x_snn  = round(x_clip / threshold * T) / T * threshold
    """

    def __init__(self, timesteps: int = 4):
        super().__init__()
        self.timesteps = int(max(1, timesteps))

    def quantize(self, x_detached: torch.Tensor, threshold: torch.Tensor, *, signed: bool) -> torch.Tensor:
        """
        Quantize a detached tensor into a low-step firing-rate approximation.
        This function is intentionally non-differentiable; use STE in the caller.
        """
        thr = torch.clamp(threshold, min=1e-8).to(dtype=x_detached.dtype, device=x_detached.device)
        t = float(self.timesteps)
        if signed:
            x_clip = torch.clamp(x_detached, -thr, thr)
            x_q = torch.round(((x_clip + thr) / (2.0 * thr)) * t)
            x_q = torch.clamp(x_q, 0.0, t)
            return (x_q / t) * (2.0 * thr) - thr
        x_clip = torch.clamp(x_detached, 0.0, thr)
        x_q = torch.round((x_clip / thr) * t) / t
        return x_q * thr

    def ste(self, x: torch.Tensor, x_quant_detached: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator: forward uses quantized, backward uses identity."""
        return x + (x_quant_detached - x).detach()
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)





class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=-1):
        super(EmbeddingNet, self).__init__()
        modules = []

        if hidden_size > 0:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SNN_EmbeddingNet(nn.Module):
    """
    Leaky integrate-and-fire stack with the same constructor shape as EmbeddingNet.
    ``use_bn`` is ignored (no BatchNorm; use floating-point affine layers only).
    Output is the time-averaged spike rate (same shape as EmbeddingNet output).
    """

    def __init__(
        self,
        input_size,
        output_size,
        dropout,
        use_bn,
        hidden_size=-1,
        num_steps=10,
        beta=0.9,
        threshold=1.0,
    ):
        super().__init__()
        if snn is None or snn_surrogate is None:
            raise ImportError(
                "SNN backend requires snntorch. Install with: pip install snntorch"
            )
        self.num_steps = int(num_steps)
        self.hidden_size = hidden_size
        spike_grad = snn_surrogate.fast_sigmoid()
        self.lif_kwargs = dict(beta=beta, threshold=threshold, spike_grad=spike_grad)

        if hidden_size > 0:
            self.lin1 = nn.Linear(input_size, hidden_size)
            self.lif1 = snn.Leaky(**self.lif_kwargs)
            self.dropout1 = nn.Dropout(dropout)
            self.lin2 = nn.Linear(hidden_size, output_size)
            self.lif2 = snn.Leaky(**self.lif_kwargs)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.lin1 = nn.Linear(input_size, output_size)
            self.lif1 = snn.Leaky(**self.lif_kwargs)
            self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        if self.hidden_size > 0:
            return self._forward_two_layer(x)
        return self._forward_one_layer(x)

    def _forward_one_layer(self, x):
        mem = torch.zeros_like(self.lin1(x))
        spike_sum = torch.zeros_like(mem)
        for _ in range(self.num_steps):
            cur = self.lin1(x)
            spk, mem = self.lif1(cur, mem)
            spk = self.dropout1(spk)
            spike_sum = spike_sum + spk
        return spike_sum / self.num_steps

    def _forward_two_layer(self, x):
        mem1 = torch.zeros_like(self.lin1(x))
        z = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(self.lin2(z))
        spike_sum = torch.zeros_like(mem2)
        for _ in range(self.num_steps):
            cur1 = self.lin1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            cur2 = self.lin2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            spike_sum = spike_sum + spk2
        return spike_sum / self.num_steps

    def get_embedding(self, x):
        return self.forward(x)


def _fuse_linear_bn1d(linear: nn.Linear, bn: nn.BatchNorm1d):
    """Fold BatchNorm1d into preceding Linear (use running stats, eval-style)."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    w = linear.weight * (gamma / std).unsqueeze(1)
    b = gamma * (linear.bias - mean) / std + beta
    return w.detach(), b.detach()


def _extract_fused_linear_weights_from_embedding_net(emb: EmbeddingNet):
    """Return list of (weight, bias) for each Linear in order, with BN fused when present."""
    mods = list(emb.fc.children())
    out = []
    n = len(mods)
    i = 0
    while i < n:
        if isinstance(mods[i], nn.Linear):
            if i + 1 < n and isinstance(mods[i + 1], nn.BatchNorm1d):
                w, b = _fuse_linear_bn1d(mods[i], mods[i + 1])
                out.append((w, b))
                i += 2
            else:
                lin = mods[i]
                out.append((lin.weight.data.clone(), lin.bias.data.clone()))
                i += 1
        else:
            i += 1
    return out


def copy_ann_embedding_net_to_snn(ann_emb: EmbeddingNet, snn_emb: "SNN_EmbeddingNet"):
    """Copy fused Linear weights from a trained EmbeddingNet into SNN_EmbeddingNet (LIF params unchanged)."""
    fused = _extract_fused_linear_weights_from_embedding_net(ann_emb)
    if snn_emb.hidden_size and snn_emb.hidden_size > 0:
        if len(fused) != 2:
            raise ValueError(
                f"Expected 2 fused Linear layers in ANN, got {len(fused)}"
            )
        snn_emb.lin1.weight.data.copy_(fused[0][0])
        snn_emb.lin1.bias.data.copy_(fused[0][1])
        snn_emb.lin2.weight.data.copy_(fused[1][0])
        snn_emb.lin2.bias.data.copy_(fused[1][1])
    else:
        if len(fused) != 1:
            raise ValueError(
                f"Expected 1 fused Linear layer in ANN, got {len(fused)}"
            )
        snn_emb.lin1.weight.data.copy_(fused[0][0])
        snn_emb.lin1.bias.data.copy_(fused[0][1])


def init_snn_clipclap_from_ann_checkpoint(
    snn_model: "ClipClap_model",
    checkpoint_path,
    device,
    model_params: dict,
    input_size_audio,
    input_size_video,
):
    """
    Load an ANN checkpoint, then copy each EmbeddingNet's fused Linear weights
    into the corresponding SNN_EmbeddingNet. LIF (beta/threshold) stays as in SNN init.
    """
    path = checkpoint_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"snn_init_ann_path not found: {path}")

    ann_params = dict(model_params)
    ann_params["model_backend"] = "ann"
    ann_params["snn_embedding_kwargs"] = {}
    ann_model = ClipClap_model(ann_params, input_size_audio, input_size_video)

    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = ann_model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"init_snn_from_ann: missing keys ({len(missing)}): {missing[:12]}")
    if unexpected:
        print(f"init_snn_from_ann: unexpected keys ({len(unexpected)}): {unexpected[:12]}")

    for name in ("O_enc", "W_enc", "O_proj", "D_o", "W_proj", "D_w"):
        ann_m = getattr(ann_model, name)
        snn_m = getattr(snn_model, name)
        if not isinstance(ann_m, EmbeddingNet) or not isinstance(snn_m, SNN_EmbeddingNet):
            raise TypeError(f"{name}: expected ANN EmbeddingNet and SNN_EmbeddingNet")
        copy_ann_embedding_net_to_snn(ann_m, snn_m)

    del ann_model
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.empty_cache()
    print("init_snn_from_ann: copied fused Linear weights from ANN checkpoint into SNN modules.")













































class ClipClap_model(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(ClipClap_model, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        self.input_dim_audio = input_size_audio
        self.input_dim_video = input_size_video

        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.drop_proj_o=params_model['dropout_decoder']
        self.drop_proj_w=params_model['additional_dropout']
        self.reg_loss=params_model['reg_loss']
        self.cross_entropy_loss=params_model['cross_entropy_loss']
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.drop_enc=params_model['dropout_encoder']


        self.rec_loss = params_model['rec_loss']

        self.lr_scheduler = params_model['lr_scheduler']

        print('Initializing trainable models...', end='')


        self.modality = params_model['modality']
        self.word_embeddings = params_model['word_embeddings']

        self._use_snn = params_model.get("model_backend") == "snn"
        Emb = SNN_EmbeddingNet if self._use_snn else EmbeddingNet
        emb_kw = params_model.get("snn_embedding_kwargs") or {}

        if self.modality == 'audio':
            self.O_enc = Emb(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True,
                **emb_kw,
            )
            self.W_enc = Emb(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True,
                **emb_kw,
            )
        elif self.modality == 'video':
            self.O_enc = Emb(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True,
                **emb_kw,
            )
            self.W_enc = Emb(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True,
                **emb_kw,
            )
        else:
            self.O_enc = Emb(
                input_size=1536,
                output_size=512,
                dropout=0.1,
                use_bn=True,
                **emb_kw,
            )
            w_in_dim = 1536
            if self.word_embeddings == 'wavcaps':
                w_in_dim = 1024
            elif self.word_embeddings == 'clip':
                w_in_dim = 512

            self.W_enc = Emb(
                input_size=w_in_dim,
                output_size=512,
                dropout=0.1,
                use_bn=True,
                **emb_kw,
            )




        word_embedding_dim = 512
        self.O_proj = Emb(
            input_size=512,
            hidden_size=self.hidden_size_decoder,
            output_size=self.dim_out,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm'],
            **emb_kw,
        )
        self.D_o = Emb(
            input_size=self.dim_out,
            hidden_size=self.hidden_size_decoder,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm'],
            **emb_kw,
        )


        self.W_proj= Emb(
            input_size=word_embedding_dim,
            output_size=self.dim_out,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm'],
            **emb_kw,
        )

        self.D_w = Emb(
            input_size=self.dim_out,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm'],
            **emb_kw,
        )









        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']

        optimizer = params_model['optimizer']
        self.is_sam_optim = False
        if optimizer == 'adam':
            self.optimizer_gen = optim.Adam(
                self.parameters(),
                lr=self.lr, weight_decay=1e-5
            )
            if self.lr_scheduler:
                self.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_gen, 'max', patience=3
                )

        elif optimizer == 'adam-sam':
            self.optimizer_gen = SAM(self.parameters(), optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                # lr scheduling on base optimizer
                self.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_gen.base_optimizer, 'max', patience=3
                )
        else:
            raise NotImplementedError

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_cyc = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        print('Done')

        # Prototype-preserving fake-SNN conversion (independent from true SNN backend).
        self.use_snn_conversion = bool(params_model.get("use_snn_conversion", False))
        self.snn_timesteps = int(params_model.get("snn_timesteps", 4))
        self.lambda_proto = float(params_model.get("lambda_proto", 1.0))
        self.lambda_feat = float(params_model.get("lambda_feat", 1.0))
        self.proto_temperature = float(params_model.get("proto_temperature", 1.0))
        self.snn_conv_threshold_percentile = float(params_model.get("snn_conv_threshold_percentile", 0.99))
        self.proto_kd_type = str(params_model.get("proto_kd_type", "kl_all"))
        self.proto_topk = int(params_model.get("proto_topk", 10))
        self.proto_warmup_epochs = int(params_model.get("proto_warmup_epochs", 0))
        self.proto_conf_margin = float(params_model.get("proto_conf_margin", 0.0))
        self.fake_snn = FakeSNNConversion(timesteps=self.snn_timesteps)

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def forward(self, a, v, w, masks, timesteps):
        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)
        if self.modality == 'audio':
            w = w[:,512:]
            model_input = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input = torch.cat((v, a), dim=1)


        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)


        rho_o = self.D_o(theta_o)


        theta_w = self.W_proj(w)


        rho_w=self.D_w(theta_w)


        output = {
            "theta_w": theta_w,
            "w": w,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "rho_o": rho_o,
        }


        return output


    def _lambda_proto_eff(self, epoch):
        """After warmup epochs, use full lambda_proto; else 0. If epoch unknown, use full weight."""
        if epoch is None:
            return float(self.lambda_proto)
        if int(epoch) < int(self.proto_warmup_epochs):
            return 0.0
        return float(self.lambda_proto)

    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy, epoch=None):

        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']


        device = theta_w.device

        theta_det = theta_o.detach()
        # theta_o distribution diagnostics (always log; fused embedding stats). Must be based on detached theta_o.
        theta_mean = theta_det.mean()
        theta_std = theta_det.std(unbiased=False)
        theta_min = theta_det.min()
        theta_max = theta_det.max()
        theta_neg_ratio = (theta_det < 0).float().mean()

        # Class prototypes (continuous) used for CE and/or prototype-preserving loss.
        embedding_cross_entropy = None
        if embeddings_crossentropy is not None:
            if self.modality == 'audio':
                embeddings_crossentropy = embeddings_crossentropy[:,512:]
            elif self.modality == 'video':
                embeddings_crossentropy = embeddings_crossentropy[:,:512]
            else:
                if self.word_embeddings == 'wavcaps':
                    embeddings_crossentropy = embeddings_crossentropy[:,512:]
                elif self.word_embeddings == 'clip':
                    embeddings_crossentropy = embeddings_crossentropy[:,:512]
            embedding_cross_entropy = self.W_proj(self.W_enc(embeddings_crossentropy))

        # Fake-SNN student embedding (fused-only). Teacher is detached theta_o.
        z_av_ann = theta_det
        z_av_snn = None
        if self.use_snn_conversion and embeddings_crossentropy is not None:
            q = float(min(1.0, max(0.5, self.snn_conv_threshold_percentile)))
            thr = torch.quantile(z_av_ann.abs().reshape(-1), q).detach()
            signed = bool(float(theta_neg_ratio.detach()) > 0.1)
            x_quant = self.fake_snn.quantize(theta_det, threshold=thr, signed=signed)
            z_av_snn = self.fake_snn.ste(theta_o, x_quant)
            # fake-SNN diagnostics (must be based on detached teacher / detached quantized)
            snn_clip_ratio = (theta_det.abs() > thr).float().mean()
            snn_zero_ratio = (x_quant.abs() < 1e-6).float().mean()
        else:
            thr = torch.tensor(0.0, device=device)
            snn_clip_ratio = torch.tensor(0.0, device=device)
            snn_zero_ratio = torch.tensor(0.0, device=device)

        # When use_snn_conversion is enabled, use z_av_snn for prediction/metrics.
        z_for_pred = z_av_snn if (self.use_snn_conversion and z_av_snn is not None) else theta_o

        if self.cross_entropy_loss==True:
            Cross_loss=nn.CrossEntropyLoss()
            scores=torch.matmul(z_for_pred, embedding_cross_entropy.t()) # (bs, 64) x (K_seen, 64).T = (bs, K_seen)
            # gt_cross_entropy = [1, 3, 2, 55, 97, 45, ...] list of gt class labels -> shape (bs,)
            l_ce=Cross_loss(scores, gt_cross_entropy)
        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(z_for_pred, theta_w)
            )
        else:
            l_reg = torch.tensor(0., device=device)


        if self.rec_loss == True:
            l_rec = (
                    self.MSE_loss(w, rho_o) +
                    self.MSE_loss(w, rho_w)
            )
        else:
            l_rec = torch.tensor(0., device=device)


        loss_original = l_rec + l_reg + l_ce
        loss_total = loss_original
        loss_proto_kd = torch.tensor(0.0, device=device)
        loss_proto_kd_weighted_tensor = torch.tensor(0.0, device=device)
        lambda_proto_eff_t = torch.tensor(0.0, device=device)
        proto_conf_keep_ratio = torch.tensor(1.0, device=device)
        proto_topk_diag = torch.tensor(float(self.proto_topk), device=device)
        feature_mse = torch.tensor(0.0, device=device)
        topk_overlap = torch.tensor(0.0, device=device)
        top1_agree = torch.tensor(0.0, device=device)

        # Prototype-preserving losses: teacher sims from theta_o.detach(); student from z_av_snn. Prototypes unchanged.
        if self.use_snn_conversion and (z_av_snn is not None):
            feature_mse = self.MSE_loss(z_av_snn, z_av_ann)

            proto = embedding_cross_entropy  # (K, dim_out)
            tau = float(max(1e-6, self.proto_temperature))
            p_n = F.normalize(proto, dim=1)
            theta_teacher_n = F.normalize(theta_o.detach(), dim=1)
            z_student_n = F.normalize(z_av_snn, dim=1)
            sim_ann = torch.matmul(theta_teacher_n, p_n.t())
            sim_snn = torch.matmul(z_student_n, p_n.t())

            n_cls = sim_ann.shape[1]
            if self.proto_conf_margin > 0.0 and n_cls >= 2:
                top2v = torch.topk(sim_ann.detach(), k=2, dim=1).values
                gap = top2v[:, 0] - top2v[:, 1]
                conf_mask = gap > self.proto_conf_margin
            else:
                conf_mask = torch.ones(sim_ann.shape[0], dtype=torch.bool, device=device)
            proto_conf_keep_ratio = conf_mask.float().mean()
            if conf_mask.any():
                sim_ann_m = sim_ann[conf_mask]
                sim_snn_m = sim_snn[conf_mask]
            else:
                sim_ann_m = sim_ann[:0]
                sim_snn_m = sim_snn[:0]

            kd_kind = self.proto_kd_type
            if kd_kind == "kl_all":
                if sim_ann_m.shape[0] == 0:
                    loss_proto_kd = torch.tensor(0.0, device=device)
                else:
                    per_s = (
                        F.kl_div(
                            F.log_softmax(sim_snn_m / tau, dim=1),
                            F.softmax(sim_ann_m.detach() / tau, dim=1),
                            reduction="none",
                        ).sum(dim=1)
                        * (tau * tau)
                    )
                    loss_proto_kd = per_s.mean()
                proto_topk_diag = torch.tensor(float(self.proto_topk), device=device)
            elif kd_kind in ("kl_topk", "mse_topk"):
                k_eff = min(self.proto_topk, n_cls)
                proto_topk_diag = torch.tensor(float(k_eff), device=device)
                if k_eff < 1 or sim_ann_m.shape[0] == 0:
                    loss_proto_kd = torch.tensor(0.0, device=device)
                else:
                    _topk_idx = torch.topk(sim_ann_m.detach(), k=k_eff, dim=1).indices
                    sim_at = torch.gather(sim_ann_m, 1, _topk_idx)
                    sim_st = torch.gather(sim_snn_m, 1, _topk_idx)
                    if kd_kind == "kl_topk":
                        per_s = (
                            F.kl_div(
                                F.log_softmax(sim_st / tau, dim=1),
                                F.softmax(sim_at.detach() / tau, dim=1),
                                reduction="none",
                            ).sum(dim=1)
                            * (tau * tau)
                        )
                        loss_proto_kd = per_s.mean()
                    else:
                        per_s = (sim_st - sim_at.detach()).pow(2).mean(dim=1)
                        loss_proto_kd = per_s.mean()
            else:
                loss_proto_kd = torch.tensor(0.0, device=device)
                proto_topk_diag = torch.tensor(float(self.proto_topk), device=device)

            # Top-k prototype consistency diagnostic (fixed k=5 as before).
            k = min(5, n_cls)
            if k > 0 and sim_ann.shape[0] > 0:
                top_ann = torch.topk(sim_ann, k=k, dim=1).indices
                top_snn = torch.topk(sim_snn, k=k, dim=1).indices
                inter = (top_ann.unsqueeze(2) == top_snn.unsqueeze(1)).any(dim=2).float().sum(dim=1)
                topk_overlap = (inter / float(k)).mean()
                top1_agree = (top_ann[:, 0] == top_snn[:, 0]).float().mean()

            lambda_eff = self._lambda_proto_eff(epoch)
            lambda_proto_eff_t = torch.tensor(lambda_eff, device=device)
            loss_proto_kd_weighted_tensor = lambda_eff * loss_proto_kd
            loss_total = loss_total + loss_proto_kd_weighted_tensor + (self.lambda_feat * feature_mse)

        loss_dict = {
            "Loss/total_loss": loss_total.detach().cpu(),
            "Loss/original_loss": loss_original.detach().cpu(),
            "Loss/loss_reg": l_reg.detach().cpu(),
            "Loss/loss_cmd_rec": l_rec.detach().cpu(),
            "Loss/cross_entropy": l_ce.detach().cpu(),
            "Loss/proto_kd": loss_proto_kd.detach().cpu(),
            "Loss/proto_kd_raw": loss_proto_kd.detach().cpu(),
            "Loss/proto_kd_weighted": loss_proto_kd_weighted_tensor.detach().cpu(),
            "Loss/feature_mse": feature_mse.detach().cpu(),
            "Diag/proto_topk_overlap": topk_overlap.detach().cpu(),
            "Diag/proto_top1_agree": top1_agree.detach().cpu(),
            "Diag/lambda_proto_eff": lambda_proto_eff_t.detach().cpu(),
            "Diag/proto_conf_keep_ratio": proto_conf_keep_ratio.detach().cpu(),
            "Diag/proto_topk": proto_topk_diag.detach().cpu(),
            "Diag/theta_o_mean": theta_mean.detach().cpu(),
            "Diag/theta_o_std": theta_std.detach().cpu(),
            "Diag/theta_o_min": theta_min.detach().cpu(),
            "Diag/theta_o_max": theta_max.detach().cpu(),
            "Diag/theta_o_negative_ratio": theta_neg_ratio.detach().cpu(),
            "Diag/snn_clip_ratio": snn_clip_ratio.detach().cpu(),
            "Diag/snn_zero_ratio": snn_zero_ratio.detach().cpu(),

        }
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False, epoch=None):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy, cls_numeric, epoch=epoch)

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy, cls_numeric, epoch=epoch)

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(outputs_second, embedding_crossentropy, cls_numeric, epoch=epoch)
                second_loss.backward()
                self.optimizer_gen.second_step(zero_grad=True)

        return loss_numeric, loss

    def get_embeddings(self, a, v, w, masks, timesteps):
        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)



        if self.modality == 'audio':
            w = w[:,512:]
            model_input = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input = torch.cat((v, a), dim=1)


        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)
        theta_w=self.W_proj(w)

        # When fake-SNN conversion is enabled, return the student embedding for similarity/metrics.
        if self.use_snn_conversion:
            theta_det = theta_o.detach()
            q = float(min(1.0, max(0.5, self.snn_conv_threshold_percentile)))
            thr = torch.quantile(theta_det.abs().reshape(-1), q).detach()
            neg_ratio = (theta_det < 0).float().mean()
            signed = bool(float(neg_ratio.detach()) > 0.1)
            x_quant = self.fake_snn.quantize(theta_det, threshold=thr, signed=signed)
            z_av_snn = self.fake_snn.ste(theta_o, x_quant)
            return z_av_snn, z_av_snn, theta_w

        return theta_o, theta_o, theta_w


def build_clipclap_model(model_params, input_size_audio, input_size_video):
    backend = model_params.get("model_backend", "ann")
    if backend not in ("ann", "snn"):
        raise ValueError(f"Unknown model_backend: {backend!r}, expected 'ann' or 'snn'.")
    return ClipClap_model(model_params, input_size_audio, input_size_video)
