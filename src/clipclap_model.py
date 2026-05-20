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


class _SpikeSurrogate(torch.autograd.Function):
    """Straight-through binary spike; backward uses 1 / (1 + |x|)^2 style surrogate."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denom = (1.0 + torch.abs(x)).pow(2).clamp(min=1e-6)
        return grad_output / denom


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


class TeacherTrainableSNNBlock(nn.Module):
    """
    Trainable two-layer LIF stack on a fixed-width vector (e.g. pooled AV).

    Full SNN route uses this over **pre-extracted** pooled audio/video features [B, D],
    not raw waveform / RGB backbone SNN.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_steps,
        beta,
        threshold,
        dropout,
        ann_dim,
        use_ann_gate,
        gate_strength,
        leak_strength,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.num_steps = int(num_steps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.use_ann_gate = bool(use_ann_gate)
        self.gate_strength = float(gate_strength)
        self.leak_strength = float(leak_strength)
        self.ann_gate = nn.Linear(ann_dim, hidden_size) if self.use_ann_gate else None
        self.scale_proj = nn.Linear(output_size, output_size)

    def forward(self, x, ann_context=None):
        B, _ = x.shape
        device, dtype = x.device, x.dtype
        h_dim = self.fc1.out_features
        out_dim = self.fc2.out_features

        gate = None
        if ann_context is not None and self.ann_gate is not None:
            gate = torch.sigmoid(self.ann_gate(ann_context))

        mem1 = torch.zeros(B, h_dim, device=device, dtype=dtype)
        mem2 = torch.zeros(B, out_dim, device=device, dtype=dtype)
        spike_count = torch.zeros(B, out_dim, device=device, dtype=dtype)
        mem2_acc = torch.zeros(B, out_dim, device=device, dtype=dtype)

        beta_s = torch.tensor(self.beta, device=device, dtype=dtype)
        thr = torch.tensor(self.threshold, device=device, dtype=dtype)

        for _ in range(self.num_steps):
            h_cur = self.fc1(x)
            if gate is not None:
                h_cur = h_cur * (1.0 - self.gate_strength * gate)

            if gate is not None:
                beta_eff = beta_s * (1.0 - self.leak_strength * gate)
                beta_eff = torch.clamp(beta_eff, 0.0, 0.99)
            else:
                beta_eff = beta_s

            mem1 = beta_eff * mem1 + h_cur
            spk1 = _SpikeSurrogate.apply(mem1 - thr)
            mem1 = mem1 * (1.0 - spk1.detach())

            out_cur = self.fc2(self.dropout(spk1))
            mem2 = beta_s * mem2 + out_cur
            spk2 = _SpikeSurrogate.apply(mem2 - thr)
            mem2 = mem2 * (1.0 - spk2.detach())

            spike_count = spike_count + spk2
            mem2_acc = mem2_acc + mem2

        n = float(self.num_steps)
        spike_rate = spike_count / n
        z = mem2_acc / n
        spike_scale = torch.sigmoid(self.scale_proj(spike_rate))

        gate_mean = gate.mean() if gate is not None else torch.tensor(0.0, device=device, dtype=dtype)
        fire_rate_mean = spike_rate.mean()

        aux = {
            "spike_count": spike_count,
            "spike_rate": spike_rate,
            "spike_scale": spike_scale,
            "gate_mean": gate_mean,
            "fire_rate_mean": fire_rate_mean,
        }
        return z, aux


class VideoTemporalSNNBranch(nn.Module):
    """Temporal LIF over pre-extracted frame features [B, T, D] -> z_video [B, out_dim]."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_steps,
        beta,
        threshold,
        dropout,
        ann_dim,
        use_ann_gate,
        gate_strength,
        leak_strength,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.num_steps = int(num_steps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.use_ann_gate = bool(use_ann_gate)
        self.gate_strength = float(gate_strength)
        self.leak_strength = float(leak_strength)
        self.ann_gate = nn.Linear(ann_dim, hidden_size) if self.use_ann_gate else None
        self.scale_proj = nn.Linear(output_size, output_size)

    def forward(self, v, ann_context=None):
        if v.dim() != 3:
            raise ValueError(f"VideoTemporalSNNBranch expects [B,T,D], got {tuple(v.shape)}")
        b, t_len, _ = v.shape
        device, dtype = v.device, v.dtype
        h_dim = self.fc1.out_features
        out_dim = self.fc2.out_features

        gate = None
        if ann_context is not None and self.ann_gate is not None:
            gate = torch.sigmoid(self.ann_gate(ann_context))

        mem1 = torch.zeros(b, h_dim, device=device, dtype=dtype)
        mem2 = torch.zeros(b, out_dim, device=device, dtype=dtype)
        spike_count = torch.zeros(b, out_dim, device=device, dtype=dtype)
        mem2_acc = torch.zeros(b, out_dim, device=device, dtype=dtype)

        beta_s = torch.tensor(self.beta, device=device, dtype=dtype)
        thr = torch.tensor(self.threshold, device=device, dtype=dtype)
        n_updates = 0.0

        for t in range(t_len):
            x_t = v[:, t, :]
            for _ in range(self.num_steps):
                h_cur = self.fc1(x_t)
                if gate is not None:
                    h_cur = h_cur * (1.0 - self.gate_strength * gate)

                if gate is not None:
                    beta_eff = beta_s * (1.0 - self.leak_strength * gate)
                    beta_eff = torch.clamp(beta_eff, 0.0, 0.99)
                else:
                    beta_eff = beta_s

                mem1 = beta_eff * mem1 + h_cur
                spk1 = _SpikeSurrogate.apply(mem1 - thr)
                mem1 = mem1 * (1.0 - spk1.detach())

                out_cur = self.fc2(self.dropout(spk1))
                mem2 = beta_s * mem2 + out_cur
                spk2 = _SpikeSurrogate.apply(mem2 - thr)
                mem2 = mem2 * (1.0 - spk2.detach())

                spike_count = spike_count + spk2
                mem2_acc = mem2_acc + mem2
                n_updates += 1.0

        n = max(n_updates, 1.0)
        spike_rate = spike_count / n
        z = mem2_acc / n
        spike_scale = torch.sigmoid(self.scale_proj(spike_rate))
        gate_mean = gate.mean() if gate is not None else torch.tensor(0.0, device=device, dtype=dtype)
        aux = {
            "spike_count": spike_count,
            "spike_rate": spike_rate,
            "spike_scale": spike_scale,
            "gate_mean": gate_mean,
            "fire_rate_mean": spike_rate.mean(),
        }
        return z, aux


class TeacherSNNFusionBranch(nn.Module):
    """Teacher SNN backend on fused latent (``TeacherTrainableSNNBlock`` + optional ANN gate)."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_steps,
        beta,
        threshold,
        dropout,
        ann_dim,
        use_ann_gate,
        gate_strength,
        leak_strength,
    ):
        super().__init__()
        self.block = TeacherTrainableSNNBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            dropout=dropout,
            ann_dim=ann_dim,
            use_ann_gate=use_ann_gate,
            gate_strength=gate_strength,
            leak_strength=leak_strength,
        )

    def forward(self, x, ann_context=None):
        return self.block(x, ann_context=ann_context)


class TeacherSNNSigmoidAnnBranch(nn.Module):
    """
    SNN-only path: pooled AV input -> z_snn [B, out_dim] and snn_gate_logits [B, out_dim].
    Gate logits come from spike_rate (no ANN input). Fusion: theta_o_refined + gamma*z_snn.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_steps,
        beta,
        threshold,
        dropout,
    ):
        super().__init__()
        self.block = TeacherTrainableSNNBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            dropout=dropout,
            ann_dim=output_size,
            use_ann_gate=False,
            gate_strength=0.0,
            leak_strength=0.0,
        )
        self.snn_int_proj = nn.Linear(output_size, output_size)

    def forward(self, x):
        z_snn, aux = self.block(x, ann_context=None)
        snn_int = self.snn_int_proj(aux["spike_rate"])
        return z_snn, snn_int, aux


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













































TEACHER_INIT_ANN_PREFIXES = ("O_enc.", "W_enc.", "O_proj.", "D_o.", "W_proj.", "D_w.")


def apply_teacher_init_ann_checkpoint(model, checkpoint_path, freeze_ann: bool, map_location):
    """
    Load matching ANN backbone tensors from a stage-1 checkpoint into ``ClipClap_model``.
    Does not load teacher SNN modules. Optionally freezes backbone and rebuilds the optimizer
    to include only ``requires_grad=True`` parameters.
    """
    path = str(checkpoint_path)
    print(f"teacher_init_ann_path: {path}", flush=True)
    ckpt = torch.load(path, map_location=map_location)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(src, dict):
        raise TypeError(f"Expected checkpoint dict or dict with 'model' key, got {type(src)}")

    tgt_state = model.state_dict()
    loaded, skipped_not_allowed, skipped_missing_in_model, skipped_shape = [], [], [], []

    for raw_k, v in src.items():
        k = raw_k.replace("module.", "", 1) if "module." in raw_k else raw_k
        if not k.startswith(TEACHER_INIT_ANN_PREFIXES):
            skipped_not_allowed.append(k)
            continue
        if k not in tgt_state:
            skipped_missing_in_model.append(k)
            continue
        if tuple(v.shape) != tuple(tgt_state[k].shape):
            skipped_shape.append(f"{k} ckpt{tuple(v.shape)} model{tuple(tgt_state[k].shape)}")
            continue
        tgt_state[k].copy_(v.to(device=tgt_state[k].device, dtype=tgt_state[k].dtype))
        loaded.append(k)

    def _fmt_keys(keys, limit=30):
        keys = list(keys)
        if len(keys) <= limit:
            return str(keys)
        return str(keys[:limit]) + f" ... (+{len(keys) - limit} more)"

    print(f"loaded ANN keys ({len(loaded)}): {_fmt_keys(loaded)}", flush=True)
    n_na, n_miss, n_shape = len(skipped_not_allowed), len(skipped_missing_in_model), len(skipped_shape)
    print(
        f"skipped keys (not_allowed={n_na}, missing_in_model={n_miss}, shape_mismatch={n_shape})",
        flush=True,
    )
    if n_na:
        print(f"  sample not_allowed: {_fmt_keys(skipped_not_allowed, limit=8)}", flush=True)
    if n_miss:
        print(f"  sample missing_in_model: {_fmt_keys(skipped_missing_in_model, limit=8)}", flush=True)
    if n_shape:
        print(f"  shape mismatches: {_fmt_keys(skipped_shape, limit=8)}", flush=True)

    if freeze_ann:
        frozen = 0
        for name, p in model.named_parameters():
            if name.startswith(TEACHER_INIT_ANN_PREFIXES):
                p.requires_grad = False
                frozen += 1
        print(f"frozen ANN params: {frozen}", flush=True)
    else:
        print("frozen ANN params: 0 (teacher_freeze_ann=False)", flush=True)

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    teacher_trainable = sum(
        1 for n, p in model.named_parameters() if p.requires_grad and not n.startswith(TEACHER_INIT_ANN_PREFIXES)
    )
    print(f"trainable teacher params: {teacher_trainable} (all trainable tensors: {trainable})", flush=True)

    model.rebuild_optimizer_trainable_only()


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

        # Teacher parallel: full ANN route (O_enc/O_proj) plus optional SNN route on the same
        # forward inputs a/v. Full SNN route (when enabled) uses trainable blocks only on
        # pre-extracted pooled AV features — not raw waveform / video backbone SNN.
        self.use_teacher_parallel_snn = bool(params_model.get("use_teacher_parallel_snn", False))
        self.teacher_snn_arch = str(params_model.get("teacher_snn_arch", "backend_only")).lower()
        if self.teacher_snn_arch not in ("backend_only", "full_snn_route", "temporal_video"):
            raise ValueError(
                "teacher_snn_arch must be 'backend_only', 'full_snn_route', or 'temporal_video', "
                f"got {self.teacher_snn_arch!r}"
            )
        self.feature_extraction_method = str(
            params_model.get("feature_extraction_method", "")
        )
        self.use_temporal_video_features = (
            self.feature_extraction_method
            in ("cls_features_temporal_16", "cls_features_static_temporal_16")
            or self.teacher_snn_arch == "temporal_video"
        )
        self.teacher_frontend_snn_timesteps = int(
            params_model.get("teacher_frontend_snn_timesteps", 4)
        )
        self.teacher_frontend_snn_hidden_dim = int(
            params_model.get("teacher_frontend_snn_hidden_dim", 512)
        )
        self.teacher_frontend_snn_decay = float(params_model.get("teacher_frontend_snn_decay", 0.9))
        self.teacher_frontend_snn_threshold = float(
            params_model.get("teacher_frontend_snn_threshold", 1.0)
        )
        self.teacher_frontend_snn_dropout = float(
            params_model.get("teacher_frontend_snn_dropout", 0.1)
        )
        self.teacher_frontend_fire_rate_target = float(
            params_model.get("teacher_frontend_fire_rate_target", 0.1)
        )
        self.teacher_frontend_fire_rate_reg = float(
            params_model.get("teacher_frontend_fire_rate_reg", 0.0)
        )
        self.teacher_snn_fusion_mode = str(
            params_model.get("teacher_snn_fusion_mode", "add")
        ).lower()
        if self.teacher_snn_fusion_mode not in ("add", "gated_scale", "snn_sigmoid_ann"):
            raise ValueError(
                "teacher_snn_fusion_mode must be 'add', 'gated_scale', or 'snn_sigmoid_ann', "
                f"got {self.teacher_snn_fusion_mode!r}"
            )
        self.teacher_sigmoid_centered = bool(params_model.get("teacher_sigmoid_centered", True))
        if self.teacher_snn_fusion_mode == "snn_sigmoid_ann":
            self.teacher_sigmoid_centered = False
        self.teacher_ann_gate_snn = bool(params_model.get("teacher_ann_gate_snn", False))
        if self.teacher_snn_fusion_mode == "snn_sigmoid_ann" and self.teacher_ann_gate_snn:
            print(
                "  warning: teacher_ann_gate_snn=True ignored for snn_sigmoid_ann (ANN must not gate SNN)",
                flush=True,
            )
            self.teacher_ann_gate_snn = False
        self._teacher_snn_ann_gate_enabled = (
            self.teacher_ann_gate_snn and self.teacher_snn_fusion_mode != "snn_sigmoid_ann"
        )
        self.teacher_freeze_ann = bool(params_model.get("teacher_freeze_ann", False))
        self.teacher_gate_strength = float(params_model.get("teacher_gate_strength", 0.5))
        self.teacher_leak_strength = float(params_model.get("teacher_leak_strength", 0.2))
        self.teacher_spike_scale_strength = float(
            params_model.get("teacher_spike_scale_strength", 0.2)
        )
        self.teacher_fire_rate_target = float(params_model.get("teacher_fire_rate_target", 0.1))
        self.teacher_fire_rate_reg = float(params_model.get("teacher_fire_rate_reg", 0.0))
        self.teacher_snn_gamma = float(params_model.get("teacher_snn_gamma", 0.1))
        self.teacher_snn_alpha = float(params_model.get("teacher_snn_alpha", 0.1))
        self.teacher_snn_beta = float(params_model.get("teacher_snn_beta", 1.0))
        self.teacher_eval_repr = str(params_model.get("teacher_eval_repr", "fused")).lower()
        if self.teacher_eval_repr not in ("ann", "snn", "fused"):
            raise ValueError(
                f"teacher_eval_repr must be 'ann', 'snn', or 'fused', got {self.teacher_eval_repr!r}"
            )
        self.teacher_fusion_warmup_epochs = int(params_model.get("teacher_fusion_warmup_epochs", 0))
        self.teacher_gamma_warmup = bool(params_model.get("teacher_gamma_warmup", True))
        self.teacher_use_snn_gate = bool(params_model.get("teacher_use_snn_gate", False))
        self.teacher_gate_mode = str(params_model.get("teacher_gate_mode", "none")).lower()
        if self.teacher_gate_mode not in ("none", "direct", "residual"):
            raise ValueError(
                f"teacher_gate_mode must be 'none', 'direct', or 'residual', got {self.teacher_gate_mode!r}"
            )
        self.teacher_sparsity_mode = str(params_model.get("teacher_sparsity_mode", "none")).lower()
        self.teacher_sparse_lambda = float(params_model.get("teacher_sparse_lambda", 0.0))
        self.use_snn_sigmoid_ann_gate = bool(params_model.get("use_snn_sigmoid_ann_gate", False))
        self.snn_sigmoid_ann_warmup_steps = int(params_model.get("snn_sigmoid_ann_warmup_steps", 500))
        self.snn_sigmoid_ann_ramp_steps = max(
            1, int(params_model.get("snn_sigmoid_ann_ramp_steps", 1500))
        )
        self.snn_sigmoid_ann_lambda = float(params_model.get("snn_sigmoid_ann_lambda", 0.3))
        self.snn_sigmoid_ann_detach_gate = bool(
            params_model.get("snn_sigmoid_ann_detach_gate", True)
        )
        self._teacher_gate_active = (
            self.use_teacher_parallel_snn
            and self.teacher_use_snn_gate
            and self.teacher_gate_mode in ("direct", "residual")
        )
        self.teacher_snn_fusion = None
        self.teacher_snn_input_size = None
        self.teacher_audio_snn_front = None
        self.teacher_video_snn_front = None
        self.teacher_video_temporal_snn = None
        self.teacher_spike_evidence_snn = None
        if self.use_teacher_parallel_snn:
            out_dim = int(self.dim_out)
            hid = int(params_model.get("teacher_snn_hidden_dim", 512))
            snn_steps = int(params_model.get("teacher_snn_timesteps", 4))
            snn_beta = float(params_model.get("teacher_snn_decay", 0.9))
            snn_thr = float(params_model.get("teacher_snn_threshold", 1.0))
            snn_drop = float(params_model.get("teacher_snn_dropout", 0.1))

            if self._teacher_gate_active:
                # Round-1: SNN only emits normalized spike evidence for sigmoid gate on theta_o.
                if self.teacher_snn_arch == "temporal_video":
                    if self.modality != "both":
                        raise ValueError("teacher gate + temporal_video requires modality=both")
                    self.teacher_video_temporal_snn = VideoTemporalSNNBranch(
                        input_size=512,
                        hidden_size=hid,
                        output_size=out_dim,
                        num_steps=snn_steps,
                        beta=snn_beta,
                        threshold=snn_thr,
                        dropout=snn_drop,
                        ann_dim=out_dim,
                        use_ann_gate=False,
                        gate_strength=0.0,
                        leak_strength=0.0,
                    )
                else:
                    if self.modality == "both":
                        fusion_in = 1536
                    elif self.modality == "audio":
                        fusion_in = 1024
                    else:
                        fusion_in = 512
                    self.teacher_snn_input_size = int(fusion_in)
                    self.teacher_spike_evidence_snn = TeacherTrainableSNNBlock(
                        input_size=self.teacher_snn_input_size,
                        hidden_size=hid,
                        output_size=out_dim,
                        num_steps=snn_steps,
                        beta=snn_beta,
                        threshold=snn_thr,
                        dropout=snn_drop,
                        ann_dim=out_dim,
                        use_ann_gate=False,
                        gate_strength=0.0,
                        leak_strength=0.0,
                    )
                print(
                    f"  teacher SNN gate mode={self.teacher_gate_mode} "
                    f"(spike_rate -> sigmoid -> refine theta_o; no SNN cls head)",
                    flush=True,
                )
            else:
                if self.teacher_snn_arch == "temporal_video":
                    if self.modality != "both":
                        raise ValueError("teacher_snn_arch=temporal_video requires modality=both")
                    self.teacher_snn_input_size = int(out_dim * 2)
                    self.teacher_video_temporal_snn = VideoTemporalSNNBranch(
                        input_size=512,
                        hidden_size=hid,
                        output_size=out_dim,
                        num_steps=snn_steps,
                        beta=snn_beta,
                        threshold=snn_thr,
                        dropout=snn_drop,
                        ann_dim=out_dim,
                        use_ann_gate=self._teacher_snn_ann_gate_enabled,
                        gate_strength=self.teacher_gate_strength,
                        leak_strength=self.teacher_leak_strength,
                    )
                    self.teacher_audio_snn_front = TeacherTrainableSNNBlock(
                        input_size=1024,
                        hidden_size=hid,
                        output_size=out_dim,
                        num_steps=snn_steps,
                        beta=snn_beta,
                        threshold=snn_thr,
                        dropout=snn_drop,
                        ann_dim=out_dim,
                        use_ann_gate=False,
                        gate_strength=0.0,
                        leak_strength=0.0,
                    )
                else:
                    if self.modality == "both":
                        fusion_in = 1536
                    elif self.modality == "audio":
                        fusion_in = 1024
                    else:
                        fusion_in = 512
                    self.teacher_snn_input_size = int(fusion_in)

                if self.teacher_snn_fusion_mode == "snn_sigmoid_ann":
                    self.teacher_snn_fusion = TeacherSNNSigmoidAnnBranch(
                        input_size=self.teacher_snn_input_size,
                        hidden_size=hid,
                        output_size=out_dim,
                        num_steps=snn_steps,
                        beta=snn_beta,
                        threshold=snn_thr,
                        dropout=snn_drop,
                    )
                else:
                    self.teacher_snn_fusion = TeacherSNNFusionBranch(
                        input_size=self.teacher_snn_input_size,
                        hidden_size=hid,
                        output_size=out_dim,
                        num_steps=snn_steps,
                        beta=snn_beta,
                        threshold=snn_thr,
                        dropout=snn_drop,
                        ann_dim=out_dim,
                        use_ann_gate=self._teacher_snn_ann_gate_enabled,
                        gate_strength=self.teacher_gate_strength,
                        leak_strength=self.teacher_leak_strength,
                    )
            if (
                not self._teacher_gate_active
                and self.teacher_snn_arch == "full_snn_route"
            ):
                fh = self.teacher_frontend_snn_hidden_dim
                fts = self.teacher_frontend_snn_timesteps
                fbeta = self.teacher_frontend_snn_decay
                fthr = self.teacher_frontend_snn_threshold
                fdrop = self.teacher_frontend_snn_dropout
                if self.modality in ("both", "audio"):
                    self.teacher_audio_snn_front = TeacherTrainableSNNBlock(
                        input_size=1024,
                        hidden_size=fh,
                        output_size=1024,
                        num_steps=fts,
                        beta=fbeta,
                        threshold=fthr,
                        dropout=fdrop,
                        ann_dim=out_dim,
                        use_ann_gate=False,
                        gate_strength=0.0,
                        leak_strength=0.0,
                    )
                if self.modality in ("both", "video"):
                    self.teacher_video_snn_front = TeacherTrainableSNNBlock(
                        input_size=512,
                        hidden_size=fh,
                        output_size=512,
                        num_steps=fts,
                        beta=fbeta,
                        threshold=fthr,
                        dropout=fdrop,
                        ann_dim=out_dim,
                        use_ann_gate=False,
                        gate_strength=0.0,
                        leak_strength=0.0,
                    )

            print(
                f"  teacher_eval_repr (val HM / get_embeddings): {self.teacher_eval_repr}",
                flush=True,
            )
            print(
                f"  teacher_fusion_warmup_epochs={self.teacher_fusion_warmup_epochs}, "
                f"teacher_gamma_warmup={self.teacher_gamma_warmup}",
                flush=True,
            )
            if self.teacher_snn_arch == "temporal_video":
                print(
                    "  teacher_snn_arch=temporal_video: VideoTemporalSNNBranch on [B,T,512], "
                    f"backend fusion in={self.teacher_snn_input_size}",
                    flush=True,
                )
            if self.teacher_snn_fusion_mode == "snn_sigmoid_ann":
                print(
                    f"  teacher_snn_fusion_mode=snn_sigmoid_ann\n"
                    f"  fusion: z_fused = theta_o_refined + gamma_eff * z_snn\n"
                    f"  theta_o_refined = (1-s)*theta_o + s*(sigmoid(snn_gate_logits)*theta_o)\n"
                    f"  use_snn_sigmoid_ann_gate={self.use_snn_sigmoid_ann_gate}\n"
                    f"  snn_sigmoid_ann_warmup_steps={self.snn_sigmoid_ann_warmup_steps}\n"
                    f"  snn_sigmoid_ann_ramp_steps={self.snn_sigmoid_ann_ramp_steps}\n"
                    f"  snn_sigmoid_ann_lambda={self.snn_sigmoid_ann_lambda}\n"
                    f"  snn_sigmoid_ann_detach_gate={self.snn_sigmoid_ann_detach_gate}\n"
                    f"  teacher_ann_gate_snn={self.teacher_ann_gate_snn}\n"
                    f"  teacher_freeze_ann={self.teacher_freeze_ann}\n"
                    f"  teacher_snn_gamma={self.teacher_snn_gamma}",
                    flush=True,
                )






        # Optimizers (only parameters with requires_grad=True; call ``rebuild_optimizer_trainable_only`` again after teacher ANN init/freeze)
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self._optimizer_name = params_model['optimizer']
        self.rebuild_optimizer_trainable_only()
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
        self.debug_print_shapes = bool(params_model.get("debug_print_shapes", False))
        self._forward_shape_debug_entry_printed = False
        self._forward_shape_debug_av_printed = False
        self._forward_shape_debug_printed = False
        self._snn_sigmoid_ann_debug_printed = False
        self.register_buffer(
            "_snn_sigmoid_ann_step",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )

    def _refine_theta_o_snn_sigmoid_ann(self, theta_o_ann, snn_gate_logits):
        """SNN sigmoid gate refines ANN theta_o (never multiplies z_snn)."""
        theta_o_refined = theta_o_ann
        gate = None
        gate_strength = torch.tensor(0.0, device=theta_o_ann.device, dtype=theta_o_ann.dtype)
        if self.use_snn_sigmoid_ann_gate:
            if self.training:
                self._snn_sigmoid_ann_step += 1
            step = self._snn_sigmoid_ann_step.float()
            warmup = float(self.snn_sigmoid_ann_warmup_steps)
            ramp = float(self.snn_sigmoid_ann_ramp_steps)
            gate_strength = ((step - warmup) / ramp).clamp(0.0, 1.0)
            gate_strength = gate_strength * float(self.snn_sigmoid_ann_lambda)
            gate = torch.sigmoid(snn_gate_logits.float())
            if self.snn_sigmoid_ann_detach_gate:
                gate = gate.detach()
            theta_o_refined = (1.0 - gate_strength) * theta_o_ann + gate_strength * (
                gate * theta_o_ann
            )
        return theta_o_refined, gate, gate_strength

    def _debug_print_snn_sigmoid_ann_once(
        self,
        theta_o,
        z_snn,
        snn_gate_logits,
        gate,
        gate_strength,
        theta_o_refined,
        z_fused,
    ):
        if not self.debug_print_shapes or self._snn_sigmoid_ann_debug_printed:
            return
        self._snn_sigmoid_ann_debug_printed = True

        def _rg(x):
            return x.requires_grad if torch.is_tensor(x) else None

        lines = [
            "[SNN-SIGMOID-ANN] use",
            f"[SNN-SIGMOID-ANN] theta_o shape={tuple(theta_o.shape)} dtype={theta_o.dtype} requires_grad={_rg(theta_o)}",
            f"[SNN-SIGMOID-ANN] z_snn shape={tuple(z_snn.shape)} dtype={z_snn.dtype} requires_grad={_rg(z_snn)}",
            f"[SNN-SIGMOID-ANN] snn_gate_logits shape={tuple(snn_gate_logits.shape)} dtype={snn_gate_logits.dtype} requires_grad={_rg(snn_gate_logits)}",
        ]
        if gate is not None:
            lines.append(
                f"[SNN-SIGMOID-ANN] gate shape={tuple(gate.shape)} dtype={gate.dtype} requires_grad={_rg(gate)}"
            )
            lines.append(
                f"[SNN-SIGMOID-ANN] gate min/mean/max={float(gate.min()):.4f}/{float(gate.mean()):.4f}/{float(gate.max()):.4f}"
            )
        else:
            lines.append("[SNN-SIGMOID-ANN] gate=None (use_snn_sigmoid_ann_gate=False)")
        gs = float(gate_strength) if torch.is_tensor(gate_strength) else gate_strength
        lines.append(f"[SNN-SIGMOID-ANN] gate_strength={gs:.6f}")
        lines.append(
            f"[SNN-SIGMOID-ANN] theta_o_refined shape={tuple(theta_o_refined.shape)} dtype={theta_o_refined.dtype} requires_grad={_rg(theta_o_refined)}"
        )
        lines.append(
            f"[SNN-SIGMOID-ANN] z_fused shape={tuple(z_fused.shape)} dtype={z_fused.dtype} requires_grad={_rg(z_fused)}"
        )
        print("\n".join(lines), flush=True)

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def rebuild_optimizer_trainable_only(self):
        """Rebuild Adam or SAM using only ``requires_grad=True`` parameters."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("rebuild_optimizer_trainable_only: no trainable parameters.")
        opt = self._optimizer_name
        self.is_sam_optim = False
        if opt == "adam":
            self.optimizer_gen = optim.Adam(trainable_params, lr=self.lr, weight_decay=1e-5)
            if self.lr_scheduler:
                self.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_gen, "max", patience=3
                )
        elif opt == "adam-sam":
            self.optimizer_gen = SAM(trainable_params, optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                self.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_gen.base_optimizer, "max", patience=3
                )
        else:
            raise NotImplementedError

    @staticmethod
    def _prepare_av_for_forward(a, v, video_static=None):
        """Pooled AV for ANN; optional [B,T,D] temporal video for teacher SNN."""
        a = a.float()
        v = v.float()
        v_seq = None
        v_static_pool = None
        if video_static is not None:
            vs = video_static.float()
            if vs.dim() == 3 and vs.shape[1] == 1:
                v_static_pool = vs.squeeze(1)
            elif vs.dim() == 2:
                v_static_pool = vs
            else:
                raise ValueError(
                    f"video_static must be [B,D] or [B,1,D], got {tuple(vs.shape)}"
                )

        if v.dim() == 3:
            if v.shape[1] == 1:
                v_pool = v.squeeze(1)
            else:
                v_seq = v
                v_pool = v_static_pool if v_static_pool is not None else v.mean(dim=1)
        elif v.dim() == 2:
            v_pool = v_static_pool if v_static_pool is not None else v
        else:
            raise ValueError(f"video must be [B,D] or [B,T,D], got {tuple(v.shape)}")

        if a.dim() == 3:
            a_pool = a.mean(dim=1) if a.shape[1] > 1 else a.squeeze(1)
        elif a.dim() == 2:
            a_pool = a
        else:
            raise ValueError(f"audio must be [B,D] or [B,T,D], got {tuple(a.shape)}")

        if v_static_pool is not None and v_seq is not None:
            ann_video_source = "static"
        elif v_seq is not None:
            ann_video_source = "temporal_mean"
        else:
            ann_video_source = "pooled"
        return a_pool, v_pool, v_seq, ann_video_source

    @staticmethod
    def _spike_evidence_for_gate(spike_count, spike_rate):
        """Use normalized spike rate (count/num_steps) as s_snn; avoids raw-count sigmoid saturation."""
        if spike_rate is not None:
            return spike_rate
        return spike_count

    def _collect_teacher_spike_evidence(self, model_input_ann, v_seq):
        """Run SNN forward for spike count/rate only (no classification head)."""
        if self.teacher_snn_arch == "temporal_video":
            if v_seq is None:
                raise ValueError(
                    "teacher gate + temporal_video requires batched video [B,T,512]; "
                    f"got no temporal sequence"
                )
            _, aux = self.teacher_video_temporal_snn(v_seq, ann_context=None)
        else:
            if self.teacher_spike_evidence_snn is None:
                raise RuntimeError("teacher_spike_evidence_snn not initialized for gate mode")
            _, aux = self.teacher_spike_evidence_snn(model_input_ann)
        return aux["spike_count"], aux["spike_rate"], aux

    def _apply_snn_gate_on_ann(self, theta_o, spike_count, spike_rate, epoch):
        """gate = sigmoid(spike_rate); refine theta_o via direct or residual gate."""
        s_snn = self._spike_evidence_for_gate(spike_count, spike_rate)
        gate = torch.sigmoid(s_snn)
        f_ann = theta_o
        gamma_eff_scalar = float(self._teacher_gamma_effective(epoch))
        if self.teacher_gate_mode == "direct":
            f_refined = gate * f_ann
        elif self.teacher_gate_mode == "residual":
            gamma_eff = torch.tensor(
                gamma_eff_scalar, device=theta_o.device, dtype=theta_o.dtype
            )
            f_refined = f_ann * (1.0 + gamma_eff * gate)
        else:
            f_refined = f_ann
        eps = 1e-8
        refine_l2_rel = ((f_refined - f_ann).norm(dim=1) / (f_ann.norm(dim=1) + eps)).mean()
        gate_diag = {
            "teacher_gate": gate,
            "teacher_gate_mean": gate.mean(),
            "teacher_gate_std": gate.std(unbiased=False),
            "teacher_gate_min": gate.min(),
            "teacher_gate_max": gate.max(),
            "teacher_refine_l2_rel": refine_l2_rel,
            "teacher_gamma_eff": torch.tensor(gamma_eff_scalar, device=theta_o.device),
        }
        return f_refined, gate, gate_diag

    def _teacher_gate_forward(self, model_input_ann, theta_o, a_pool, v_pool, v_seq, epoch):
        """SNN spike evidence -> sigmoid gate -> refine ANN theta_o (Step 2)."""
        empty = {
            "teacher_model_input_ann": None,
            "teacher_model_input_snn": None,
            "teacher_theta_ann": None,
            "teacher_z_snn": None,
            "teacher_z_fused": None,
            "teacher_theta_scaled": None,
            "teacher_spike_count": None,
            "teacher_spike_rate": None,
            "teacher_spike_scale": None,
            "teacher_gate_mean": None,
            "teacher_fire_rate_mean": None,
            "teacher_fusion_mode": None,
            "teacher_gate_active": True,
            "teacher_gate": None,
            "teacher_gate_std": None,
            "teacher_gate_min": None,
            "teacher_gate_max": None,
            "teacher_refine_l2_rel": None,
            "teacher_gamma_eff": None,
            "teacher_audio_snn_front": None,
            "teacher_video_snn_front": None,
            "teacher_video_temporal_z": None,
            "teacher_audio_front_spike_count": None,
            "teacher_audio_front_spike_rate": None,
            "teacher_audio_front_spike_scale": None,
            "teacher_audio_front_fire_rate_mean": None,
            "teacher_video_front_spike_count": None,
            "teacher_video_front_spike_rate": None,
            "teacher_video_front_spike_scale": None,
            "teacher_video_front_fire_rate_mean": None,
        }
        spike_count, spike_rate, aux = self._collect_teacher_spike_evidence(
            model_input_ann, v_seq
        )
        f_refined, gate, gate_diag = self._apply_snn_gate_on_ann(
            theta_o, spike_count, spike_rate, epoch
        )
        teacher_z_snn_diag = f_refined - theta_o
        out = dict(empty)
        out.update(
            {
                "teacher_model_input_ann": model_input_ann,
                "teacher_theta_ann": theta_o,
                "teacher_z_snn": teacher_z_snn_diag,
                "teacher_z_fused": f_refined,
                "teacher_spike_count": spike_count,
                "teacher_spike_rate": spike_rate,
                "teacher_fire_rate_mean": aux.get("fire_rate_mean"),
                "teacher_fusion_mode": self.teacher_gate_mode,
                "teacher_gate": gate_diag["teacher_gate"],
                "teacher_gate_mean": gate_diag["teacher_gate_mean"],
                "teacher_gate_std": gate_diag["teacher_gate_std"],
                "teacher_gate_min": gate_diag["teacher_gate_min"],
                "teacher_gate_max": gate_diag["teacher_gate_max"],
                "teacher_refine_l2_rel": gate_diag["teacher_refine_l2_rel"],
                "teacher_gamma_eff": gate_diag["teacher_gamma_eff"],
            }
        )
        if self.teacher_snn_arch == "temporal_video":
            out["teacher_video_front_spike_count"] = spike_count
            out["teacher_video_front_spike_rate"] = spike_rate
            out["teacher_video_front_fire_rate_mean"] = aux.get("fire_rate_mean")
        return out

    def _teacher_parallel_forward(self, model_input_ann, theta_o, a_pool, v_pool, v_seq, epoch):
        """Run teacher SNN branch; returns dict of teacher outputs (or Nones if disabled)."""
        empty = {
            "teacher_gate_active": False,
            "teacher_model_input_ann": None,
            "teacher_model_input_snn": None,
            "teacher_theta_ann": None,
            "teacher_z_snn": None,
            "teacher_z_fused": None,
            "teacher_theta_scaled": None,
            "teacher_spike_count": None,
            "teacher_spike_rate": None,
            "teacher_spike_scale": None,
            "teacher_gate_mean": None,
            "teacher_fire_rate_mean": None,
            "teacher_fusion_mode": None,
            "teacher_audio_snn_front": None,
            "teacher_video_snn_front": None,
            "teacher_video_temporal_z": None,
            "teacher_audio_front_spike_count": None,
            "teacher_audio_front_spike_rate": None,
            "teacher_audio_front_spike_scale": None,
            "teacher_audio_front_fire_rate_mean": None,
            "teacher_video_front_spike_count": None,
            "teacher_video_front_spike_rate": None,
            "teacher_video_front_spike_scale": None,
            "teacher_video_front_fire_rate_mean": None,
            "teacher_gate": None,
            "teacher_gate_std": None,
            "teacher_gate_min": None,
            "teacher_gate_max": None,
            "teacher_refine_l2_rel": None,
            "teacher_gamma_eff": None,
            "teacher_theta_o_ann": None,
            "teacher_theta_o_refined": None,
            "teacher_snn_gate_strength": None,
            "teacher_snn_gate_logits": None,
        }
        if self._teacher_gate_active:
            return self._teacher_gate_forward(
                model_input_ann, theta_o, a_pool, v_pool, v_seq, epoch
            )
        if not (self.use_teacher_parallel_snn and self.teacher_snn_fusion is not None):
            return empty

        teacher_model_input_ann = model_input_ann
        teacher_theta_ann = theta_o
        aux_a = None
        aux_v = None
        teacher_audio_snn_front = None
        teacher_video_snn_front = None
        teacher_video_temporal_z = None

        if self.teacher_snn_arch == "temporal_video":
            if v_seq is None:
                raise ValueError(
                    "teacher_snn_arch=temporal_video requires batched video [B,T,512]; "
                    f"got pooled shape {tuple(v_pool.shape)}"
                )
            ann_ctx = theta_o if self._teacher_snn_ann_gate_enabled else None
            teacher_video_temporal_z, aux_v = self.teacher_video_temporal_snn(v_seq, ann_context=ann_ctx)
            z_audio_snn, aux_a = self.teacher_audio_snn_front(a_pool)
            teacher_audio_snn_front = z_audio_snn
            model_input_snn = torch.cat((teacher_video_temporal_z, z_audio_snn), dim=1)
        elif self.teacher_snn_arch == "full_snn_route":
            if self.modality == "both":
                a_snn, aux_a = self.teacher_audio_snn_front(a_pool)
                v_snn, aux_v = self.teacher_video_snn_front(v_pool)
                teacher_audio_snn_front = a_snn
                teacher_video_snn_front = v_snn
                model_input_snn = torch.cat((v_snn, a_snn), dim=1)
            elif self.modality == "audio":
                a_snn, aux_a = self.teacher_audio_snn_front(a_pool)
                teacher_audio_snn_front = a_snn
                model_input_snn = a_snn
            else:
                v_snn, aux_v = self.teacher_video_snn_front(v_pool)
                teacher_video_snn_front = v_snn
                model_input_snn = v_snn
        else:
            model_input_snn = model_input_ann

        assert model_input_snn.shape[1] == self.teacher_snn_input_size, (
            f"Teacher SNN backend input dim mismatch: "
            f"model_input_snn has {model_input_snn.shape[1]}, "
            f"backend expects {self.teacher_snn_input_size}"
        )
        gamma_eff = self._teacher_gamma_effective(epoch)
        teacher_fusion_mode = self.teacher_snn_fusion_mode
        teacher_gate = None
        teacher_snn_int = None
        teacher_theta_o_ann = None
        teacher_theta_o_refined = None
        teacher_snn_gate_strength = None
        teacher_snn_gate_logits = None
        if self.teacher_snn_fusion_mode == "snn_sigmoid_ann":
            z_snn, snn_int, teacher_aux = self.teacher_snn_fusion(model_input_snn)
            snn_gate_logits = snn_int
            teacher_snn_int = snn_int
            teacher_snn_gate_logits = snn_gate_logits
            theta_o_ann = theta_o
            teacher_theta_o_ann = theta_o_ann
            theta_o_refined, teacher_gate, teacher_snn_gate_strength = (
                self._refine_theta_o_snn_sigmoid_ann(theta_o_ann, snn_gate_logits)
            )
            teacher_theta_o_refined = theta_o_refined
            teacher_z_fused = theta_o_refined + gamma_eff * z_snn
            teacher_z_snn = z_snn
            teacher_theta_scaled = None
            self._debug_print_snn_sigmoid_ann_once(
                theta_o_ann,
                z_snn,
                snn_gate_logits,
                teacher_gate,
                teacher_snn_gate_strength,
                theta_o_refined,
                teacher_z_fused,
            )
        else:
            ann_ctx = theta_o if self._teacher_snn_ann_gate_enabled else None
            teacher_z_snn, teacher_aux = self.teacher_snn_fusion(
                model_input_snn, ann_context=ann_ctx
            )
            if self.teacher_snn_fusion_mode == "gated_scale":
                teacher_theta_scaled = theta_o * (
                    1.0
                    + self.teacher_spike_scale_strength
                    * (2.0 * teacher_aux["spike_scale"] - 1.0)
                )
                teacher_z_fused = teacher_theta_scaled + gamma_eff * teacher_z_snn
            else:
                teacher_theta_scaled = None
                teacher_z_fused = theta_o + gamma_eff * teacher_z_snn

        out = dict(empty)
        out.update(
            {
                "teacher_model_input_ann": teacher_model_input_ann,
                "teacher_model_input_snn": model_input_snn,
                "teacher_theta_ann": teacher_theta_ann,
                "teacher_z_snn": teacher_z_snn,
                "teacher_z_fused": teacher_z_fused,
                "teacher_theta_scaled": teacher_theta_scaled,
                "teacher_spike_count": teacher_aux["spike_count"],
                "teacher_spike_rate": teacher_aux["spike_rate"],
                "teacher_spike_scale": teacher_aux["spike_scale"],
                "teacher_gate_mean": teacher_aux["gate_mean"],
                "teacher_fire_rate_mean": teacher_aux["fire_rate_mean"],
                "teacher_fusion_mode": teacher_fusion_mode,
                "teacher_gate": teacher_gate,
                "teacher_snn_int": teacher_snn_int,
                "teacher_theta_o_ann": teacher_theta_o_ann,
                "teacher_theta_o_refined": teacher_theta_o_refined,
                "teacher_snn_gate_strength": teacher_snn_gate_strength,
                "teacher_snn_gate_logits": teacher_snn_gate_logits,
                "teacher_gamma_eff": gamma_eff,
                "teacher_audio_snn_front": teacher_audio_snn_front,
                "teacher_video_snn_front": teacher_video_snn_front,
                "teacher_video_temporal_z": teacher_video_temporal_z,
            }
        )
        if aux_a is not None:
            out["teacher_audio_front_spike_count"] = aux_a["spike_count"]
            out["teacher_audio_front_spike_rate"] = aux_a["spike_rate"]
            out["teacher_audio_front_spike_scale"] = aux_a["spike_scale"]
            out["teacher_audio_front_fire_rate_mean"] = aux_a["fire_rate_mean"]
        if aux_v is not None and self.teacher_snn_arch == "full_snn_route":
            out["teacher_video_front_spike_count"] = aux_v["spike_count"]
            out["teacher_video_front_spike_rate"] = aux_v["spike_rate"]
            out["teacher_video_front_spike_scale"] = aux_v["spike_scale"]
            out["teacher_video_front_fire_rate_mean"] = aux_v["fire_rate_mean"]
        elif aux_v is not None and self.teacher_snn_arch == "temporal_video":
            out["teacher_video_front_spike_count"] = aux_v["spike_count"]
            out["teacher_video_front_spike_rate"] = aux_v["spike_rate"]
            out["teacher_video_front_spike_scale"] = aux_v["spike_scale"]
            out["teacher_video_front_fire_rate_mean"] = aux_v["fire_rate_mean"]
        return out

    def forward(self, a, v, w, masks, timesteps, epoch=None, video_static=None):
        w_cls_raw = w

        def _shape_desc(tag, x, lines_out):
            if torch.is_tensor(x):
                lines_out.append(f"  {tag}: Tensor shape={tuple(x.shape)} dtype={x.dtype}")
            elif isinstance(x, np.ndarray):
                lines_out.append(f"  {tag}: ndarray shape={x.shape} dtype={x.dtype}")
            elif isinstance(x, dict):
                lines_out.append(f"  {tag}: dict keys={list(x.keys())}")
                for k, v in x.items():
                    _shape_desc(f"{tag}.{k}", v, lines_out)
            elif isinstance(x, (list, tuple)):
                lines_out.append(f"  {tag}: {type(x).__name__} len={len(x)}")
            else:
                lines_out.append(f"  {tag}: {type(x).__name__}")

        if self.debug_print_shapes and not self._forward_shape_debug_entry_printed:
            self._forward_shape_debug_entry_printed = True
            pre_lines = [
                "[ClipClap_model.forward] one-time debug (BEFORE b, _ = a.shape / torch.cat / O_enc):",
            ]
            _shape_desc("a (raw forward input)", a, pre_lines)
            _shape_desc("v (raw temporal video)", v, pre_lines)
            _shape_desc("video_static", video_static, pre_lines)
            _shape_desc("w (forward arg cls_embedding, raw)", w_cls_raw, pre_lines)
            _shape_desc("masks", masks, pre_lines)
            _shape_desc("timesteps", timesteps, pre_lines)
            print("\n".join(pre_lines), flush=True)

        a_pool, v_pool, v_seq, ann_video_source = self._prepare_av_for_forward(
            a, v, video_static=video_static
        )
        b = a_pool.shape[0]
        device = a_pool.device

        if self.debug_print_shapes and not self._forward_shape_debug_av_printed:
            self._forward_shape_debug_av_printed = True
            av_lines = [
                "[ClipClap_model.forward] AV routing (after _prepare_av_for_forward):",
                f"  v raw temporal shape: {tuple(v.shape)}",
                f"  video_static shape: {tuple(video_static.shape) if video_static is not None else None}",
                f"  ANN video source: {ann_video_source}",
                f"  v_pool (ANN video) shape: {tuple(v_pool.shape)}",
            ]
            if v_seq is not None:
                av_lines.append(f"  teacher video SNN input shape: {tuple(v_seq.shape)}")
            else:
                av_lines.append("  teacher video SNN input: None (no temporal sequence)")
            print("\n".join(av_lines), flush=True)

        if self.modality == 'audio':
            w = w[:,512:]
            model_input_ann = a_pool

        elif self.modality == 'video':
            w = w[:,:512]
            model_input_ann = v_pool
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input_ann = torch.cat((v_pool, a_pool), dim=1)

        model_input = model_input_ann

        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)


        rho_o = self.D_o(theta_o)


        theta_w = self.W_proj(w)

        t_out = self._teacher_parallel_forward(
            model_input_ann, theta_o, a_pool, v_pool, v_seq, epoch
        )
        teacher_z_snn = t_out["teacher_z_snn"]
        teacher_z_fused = t_out["teacher_z_fused"]
        teacher_theta_scaled = t_out["teacher_theta_scaled"]
        teacher_spike_count = t_out["teacher_spike_count"]
        teacher_spike_rate = t_out["teacher_spike_rate"]
        teacher_spike_scale = t_out["teacher_spike_scale"]
        teacher_gate_mean = t_out["teacher_gate_mean"]
        teacher_fire_rate_mean = t_out["teacher_fire_rate_mean"]
        teacher_fusion_mode = t_out["teacher_fusion_mode"]
        teacher_model_input_ann = t_out["teacher_model_input_ann"]
        teacher_model_input_snn = t_out["teacher_model_input_snn"]
        teacher_theta_ann = t_out["teacher_theta_ann"]
        teacher_audio_snn_front = t_out["teacher_audio_snn_front"]
        teacher_video_snn_front = t_out["teacher_video_snn_front"]
        teacher_audio_front_spike_count = t_out["teacher_audio_front_spike_count"]
        teacher_audio_front_spike_rate = t_out["teacher_audio_front_spike_rate"]
        teacher_audio_front_spike_scale = t_out["teacher_audio_front_spike_scale"]
        teacher_audio_front_fire_rate_mean = t_out["teacher_audio_front_fire_rate_mean"]
        teacher_video_front_spike_count = t_out["teacher_video_front_spike_count"]
        teacher_video_front_spike_rate = t_out["teacher_video_front_spike_rate"]
        teacher_video_front_spike_scale = t_out["teacher_video_front_spike_scale"]
        teacher_video_front_fire_rate_mean = t_out["teacher_video_front_fire_rate_mean"]

        if self.debug_print_shapes and not self._forward_shape_debug_printed:
            self._forward_shape_debug_printed = True
            post_lines = [
                "[ClipClap_model.forward] one-time debug (AFTER model_input / O_enc / O_proj / theta_w):",
            ]
            _shape_desc("model_input (ANN route)", model_input, post_lines)
            _shape_desc("o", o, post_lines)
            _shape_desc("theta_o", theta_o, post_lines)
            _shape_desc("theta_w", theta_w, post_lines)
            if self.use_teacher_parallel_snn:
                assert teacher_z_snn is not None and teacher_z_fused is not None, (
                    "use_teacher_parallel_snn=True but teacher_z_snn/teacher_z_fused missing"
                )
                _shape_desc("teacher_model_input_ann", teacher_model_input_ann, post_lines)
                _shape_desc("teacher_theta_ann", teacher_theta_ann, post_lines)
                if self.teacher_snn_arch in ("full_snn_route", "temporal_video"):
                    _shape_desc("teacher_audio_snn_front", teacher_audio_snn_front, post_lines)
                    _shape_desc("teacher_video_snn_front", teacher_video_snn_front, post_lines)
                    _shape_desc("teacher_model_input_snn", teacher_model_input_snn, post_lines)
                    _shape_desc("teacher_audio_front_spike_rate", teacher_audio_front_spike_rate, post_lines)
                    _shape_desc("teacher_video_front_spike_rate", teacher_video_front_spike_rate, post_lines)
                else:
                    _shape_desc("teacher_model_input_snn", teacher_model_input_snn, post_lines)
                _shape_desc(
                    "teacher_z_snn (SNN backend on pooled AV latent)",
                    teacher_z_snn,
                    post_lines,
                )
                _shape_desc("teacher_z_fused", teacher_z_fused, post_lines)
                if teacher_theta_scaled is not None:
                    _shape_desc("teacher_theta_scaled", teacher_theta_scaled, post_lines)
                _shape_desc("teacher_spike_rate (backend)", teacher_spike_rate, post_lines)
                _shape_desc("teacher_spike_count (backend)", teacher_spike_count, post_lines)
                _shape_desc("teacher_spike_scale (backend)", teacher_spike_scale, post_lines)
                _shape_desc("teacher_gate_mean", teacher_gate_mean, post_lines)
                _shape_desc("teacher_fire_rate_mean (backend)", teacher_fire_rate_mean, post_lines)
            print("\n".join(post_lines), flush=True)

        rho_w=self.D_w(theta_w)


        output = {
            "theta_w": theta_w,
            "w": w,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "rho_o": rho_o,
            "teacher_snn_arch": self.teacher_snn_arch if self.use_teacher_parallel_snn else None,
            "teacher_model_input_ann": teacher_model_input_ann,
            "teacher_model_input_snn": teacher_model_input_snn,
            "teacher_theta_ann": teacher_theta_ann,
            "teacher_z_snn": teacher_z_snn,
            "teacher_z_fused": teacher_z_fused,
            "teacher_theta_scaled": teacher_theta_scaled,
            "teacher_spike_count": teacher_spike_count,
            "teacher_spike_rate": teacher_spike_rate,
            "teacher_spike_scale": teacher_spike_scale,
            "teacher_gate_mean": teacher_gate_mean,
            "teacher_fire_rate_mean": teacher_fire_rate_mean,
            "teacher_fusion_mode": teacher_fusion_mode,
            "teacher_audio_snn_front": teacher_audio_snn_front,
            "teacher_video_snn_front": teacher_video_snn_front,
            "teacher_audio_front_spike_count": teacher_audio_front_spike_count,
            "teacher_audio_front_spike_rate": teacher_audio_front_spike_rate,
            "teacher_audio_front_spike_scale": teacher_audio_front_spike_scale,
            "teacher_audio_front_fire_rate_mean": teacher_audio_front_fire_rate_mean,
            "teacher_video_front_spike_count": teacher_video_front_spike_count,
            "teacher_video_front_spike_rate": teacher_video_front_spike_rate,
            "teacher_video_front_spike_scale": teacher_video_front_spike_scale,
            "teacher_video_front_fire_rate_mean": teacher_video_front_fire_rate_mean,
            "teacher_parallel_enabled": bool(
                self._teacher_gate_active
                or (self.use_teacher_parallel_snn and self.teacher_snn_fusion is not None)
            ),
            "teacher_gate_active": bool(self._teacher_gate_active),
            "teacher_gate": t_out.get("teacher_gate"),
            "teacher_gate_std": t_out.get("teacher_gate_std"),
            "teacher_gate_min": t_out.get("teacher_gate_min"),
            "teacher_gate_max": t_out.get("teacher_gate_max"),
            "teacher_refine_l2_rel": t_out.get("teacher_refine_l2_rel"),
            "teacher_gamma_eff": t_out.get("teacher_gamma_eff"),
            "teacher_snn_int": t_out.get("teacher_snn_int"),
            "teacher_theta_o_ann": t_out.get("teacher_theta_o_ann"),
            "teacher_theta_o_refined": t_out.get("teacher_theta_o_refined"),
            "teacher_snn_gate_strength": t_out.get("teacher_snn_gate_strength"),
            "teacher_snn_gate_logits": t_out.get("teacher_snn_gate_logits"),
        }


        return output


    def _lambda_proto_eff(self, epoch):
        """After warmup epochs, use full lambda_proto; else 0. If epoch unknown, use full weight."""
        if epoch is None:
            return float(self.lambda_proto)
        if int(epoch) < int(self.proto_warmup_epochs):
            return 0.0
        return float(self.lambda_proto)

    def _teacher_gamma_effective(self, epoch):
        """Gamma for legacy z_snn fusion or gate-residual modulation on theta_o; epoch None => full gamma."""
        g = float(self.teacher_snn_gamma)
        if not self.use_teacher_parallel_snn:
            return g
        if epoch is None:
            return g
        if self.teacher_gamma_warmup and self.teacher_fusion_warmup_epochs > 0:
            return g * min(1.0, float(epoch) / float(self.teacher_fusion_warmup_epochs))
        return g

    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy, epoch=None):

        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']

        teacher_z_snn = outputs.get("teacher_z_snn")
        teacher_z_fused = outputs.get("teacher_z_fused")

        device = theta_w.device
        loss_teacher_fire = None
        loss_teacher_front_fire = None

        l_ann_teacher = torch.tensor(0.0, device=device)
        l_snn_teacher = torch.tensor(0.0, device=device)
        l_fused_teacher = torch.tensor(0.0, device=device)
        teacher_parallel_ce_used = False
        teacher_gate_ce_used = False

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
            use_teacher_gate_ce = (
                outputs.get("teacher_gate_active")
                and teacher_z_fused is not None
                and embedding_cross_entropy is not None
            )
            use_teacher_parallel_ce = (
                self.use_teacher_parallel_snn
                and outputs.get("teacher_parallel_enabled")
                and not outputs.get("teacher_gate_active")
                and teacher_z_snn is not None
                and teacher_z_fused is not None
                and embedding_cross_entropy is not None
            )
            if embedding_cross_entropy is None:
                l_ce = torch.tensor(0., device=device)
            elif use_teacher_gate_ce:
                teacher_gate_ce_used = True
                teacher_parallel_ce_used = True

                def _teacher_ce_logits(z):
                    return torch.matmul(z, embedding_cross_entropy.t())

                l_fused_teacher = Cross_loss(
                    _teacher_ce_logits(teacher_z_fused), gt_cross_entropy
                )
                l_ann_teacher = Cross_loss(_teacher_ce_logits(theta_o), gt_cross_entropy)
                l_ce = l_fused_teacher + self.teacher_snn_beta * l_ann_teacher
            elif use_teacher_parallel_ce:
                teacher_parallel_ce_used = True

                def _teacher_ce_logits(z):
                    return torch.matmul(z, embedding_cross_entropy.t())

                l_ann_teacher = Cross_loss(_teacher_ce_logits(theta_o), gt_cross_entropy)
                l_fused_teacher = Cross_loss(_teacher_ce_logits(teacher_z_fused), gt_cross_entropy)
                if self.teacher_snn_fusion_mode == "snn_sigmoid_ann":
                    l_snn_teacher = torch.tensor(0.0, device=device)
                    l_ce = l_fused_teacher + self.teacher_snn_beta * l_ann_teacher
                else:
                    l_snn_teacher = Cross_loss(
                        _teacher_ce_logits(teacher_z_snn), gt_cross_entropy
                    )
                    l_ce = (
                        l_fused_teacher
                        + self.teacher_snn_alpha * l_snn_teacher
                        + self.teacher_snn_beta * l_ann_teacher
                    )
            else:
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
        if (
            self.use_teacher_parallel_snn
            and not outputs.get("teacher_gate_active")
            and self.teacher_fire_rate_reg > 0.0
            and self.teacher_sparse_lambda <= 0.0
            and outputs.get("teacher_spike_rate") is not None
        ):
            sr = outputs["teacher_spike_rate"]
            loss_teacher_fire = ((sr - self.teacher_fire_rate_target) ** 2).mean()
            loss_original = loss_original + self.teacher_fire_rate_reg * loss_teacher_fire

        if (
            self.use_teacher_parallel_snn
            and not outputs.get("teacher_gate_active")
            and self.teacher_snn_arch in ("full_snn_route", "temporal_video")
            and self.teacher_frontend_fire_rate_reg > 0.0
        ):
            parts = []
            ar = outputs.get("teacher_audio_front_spike_rate")
            vr = outputs.get("teacher_video_front_spike_rate")
            tar = self.teacher_frontend_fire_rate_target
            if ar is not None:
                parts.append(((ar - tar) ** 2).mean())
            if vr is not None:
                parts.append(((vr - tar) ** 2).mean())
            if parts:
                loss_teacher_front_fire = torch.stack(parts).mean()
                loss_original = (
                    loss_original
                    + self.teacher_frontend_fire_rate_reg * loss_teacher_front_fire
                )

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
        if teacher_parallel_ce_used:
            loss_dict["Loss/loss_teacher_ann"] = l_ann_teacher.detach().cpu()
            loss_dict["Loss/loss_teacher_fused"] = l_fused_teacher.detach().cpu()
            loss_dict["Loss/loss_teacher_ce"] = l_ce.detach().cpu()
            loss_dict["Diag/teacher_snn_gamma"] = torch.tensor(float(self.teacher_snn_gamma))
            loss_dict["Diag/teacher_snn_beta"] = torch.tensor(float(self.teacher_snn_beta))
            if teacher_gate_ce_used:
                loss_dict["Loss/loss_teacher_snn"] = torch.tensor(0.0)
                loss_dict["Diag/teacher_gate_ce_used"] = torch.tensor(1.0)
            else:
                loss_dict["Loss/loss_teacher_snn"] = l_snn_teacher.detach().cpu()
                loss_dict["Diag/teacher_gate_ce_used"] = torch.tensor(0.0)
                loss_dict["Diag/teacher_snn_alpha"] = torch.tensor(float(self.teacher_snn_alpha))
        if outputs.get("teacher_gate_active"):
            if outputs.get("teacher_gate_mean") is not None:
                loss_dict["Diag/gate_mean"] = outputs["teacher_gate_mean"].detach().cpu()
            if outputs.get("teacher_gate_std") is not None:
                loss_dict["Diag/gate_std"] = outputs["teacher_gate_std"].detach().cpu()
            if outputs.get("teacher_gate_min") is not None:
                loss_dict["Diag/gate_min"] = outputs["teacher_gate_min"].detach().cpu()
            if outputs.get("teacher_gate_max") is not None:
                loss_dict["Diag/gate_max"] = outputs["teacher_gate_max"].detach().cpu()
            if outputs.get("teacher_spike_rate") is not None:
                loss_dict["Diag/spike_rate_mean"] = outputs["teacher_spike_rate"].mean().detach().cpu()
            if outputs.get("teacher_spike_count") is not None:
                loss_dict["Diag/spike_count_mean"] = outputs["teacher_spike_count"].mean().detach().cpu()
            if outputs.get("teacher_refine_l2_rel") is not None:
                loss_dict["Diag/refine_l2_rel"] = outputs["teacher_refine_l2_rel"].detach().cpu()
            if outputs.get("teacher_gamma_eff") is not None:
                loss_dict["Diag/gamma_eff"] = outputs["teacher_gamma_eff"].detach().cpu()
            loss_dict["Diag/teacher_gate_mode_direct"] = torch.tensor(
                1.0 if self.teacher_gate_mode == "direct" else 0.0
            )
        if (
            outputs.get("teacher_parallel_enabled")
            and outputs.get("teacher_spike_rate") is not None
            and not outputs.get("teacher_gate_active")
        ):
            loss_dict["Diag/teacher_fire_rate_mean"] = outputs["teacher_fire_rate_mean"].detach().cpu()
            loss_dict["Diag/teacher_gate_mean"] = outputs["teacher_gate_mean"].detach().cpu()
            loss_dict["Diag/teacher_spike_scale_mean"] = outputs["teacher_spike_scale"].mean().detach().cpu()
            fusion_mode = str(outputs.get("teacher_fusion_mode") or "")
            loss_dict["Diag/teacher_fusion_mode_gated"] = torch.tensor(
                1.0 if fusion_mode == "gated_scale" else 0.0
            )
            loss_dict["Diag/teacher_fusion_mode_snn_sigmoid_ann"] = torch.tensor(
                1.0 if fusion_mode == "snn_sigmoid_ann" else 0.0
            )
            if outputs.get("teacher_gate") is not None:
                loss_dict["Diag/teacher_gate_mean"] = outputs["teacher_gate"].mean().detach().cpu()
            if outputs.get("teacher_snn_gate_strength") is not None:
                loss_dict["Diag/snn_sigmoid_ann_gate_strength"] = (
                    outputs["teacher_snn_gate_strength"].detach().cpu()
                )
            if outputs.get("teacher_gamma_eff") is not None:
                loss_dict["Diag/teacher_gamma_eff_forward"] = torch.tensor(
                    float(outputs["teacher_gamma_eff"])
                )
            if loss_teacher_fire is not None:
                loss_dict["Loss/loss_teacher_fire_rate"] = loss_teacher_fire.detach().cpu()
            else:
                loss_dict["Loss/loss_teacher_fire_rate"] = torch.tensor(0.0)
        if (
            outputs.get("teacher_parallel_enabled")
            and not outputs.get("teacher_gate_active")
            and str(outputs.get("teacher_snn_arch") or "") in ("full_snn_route", "temporal_video")
        ):
            if loss_teacher_front_fire is not None:
                loss_dict["Loss/loss_teacher_front_fire"] = loss_teacher_front_fire.detach().cpu()
            else:
                loss_dict["Loss/loss_teacher_front_fire"] = torch.tensor(0.0)
            if outputs.get("teacher_audio_front_fire_rate_mean") is not None:
                loss_dict["Diag/teacher_audio_front_fire_rate_mean"] = (
                    outputs["teacher_audio_front_fire_rate_mean"].detach().cpu()
                )
            if outputs.get("teacher_video_front_fire_rate_mean") is not None:
                loss_dict["Diag/teacher_video_front_fire_rate_mean"] = (
                    outputs["teacher_video_front_fire_rate_mean"].detach().cpu()
                )
        if self.use_teacher_parallel_snn:
            loss_dict["Diag/teacher_gamma_eff"] = torch.tensor(
                float(self._teacher_gamma_effective(epoch))
            )
            loss_dict["Diag/teacher_fusion_warmup_epochs"] = torch.tensor(
                float(self.teacher_fusion_warmup_epochs)
            )
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(
        self,
        audio,
        video,
        cls_numeric,
        cls_embedding,
        masks,
        timesteps,
        embedding_crossentropy,
        optimize=False,
        epoch=None,
        video_static=None,
    ):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(
                audio,
                video,
                cls_embedding,
                masks,
                timesteps,
                epoch=epoch,
                video_static=video_static,
            )

            # Backward pass
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy, cls_numeric, epoch=epoch)

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(
                audio,
                video,
                cls_embedding,
                masks,
                timesteps,
                epoch=epoch,
                video_static=video_static,
            )
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy, cls_numeric, epoch=epoch)

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(
                    audio,
                    video,
                    cls_embedding,
                    masks,
                    timesteps,
                    epoch=epoch,
                    video_static=video_static,
                )
                second_loss, _ = self.compute_loss(outputs_second, embedding_crossentropy, cls_numeric, epoch=epoch)
                second_loss.backward()
                self.optimizer_gen.second_step(zero_grad=True)

        return loss_numeric, loss

    def get_embeddings(self, a, v, w, masks, timesteps, video_static=None):
        a_pool, v_pool, v_seq, _ann_src = self._prepare_av_for_forward(
            a, v, video_static=video_static
        )

        if self.modality == 'audio':
            w = w[:,512:]
            model_input_ann = a_pool

        elif self.modality == 'video':
            w = w[:,:512]
            model_input_ann = v_pool
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input_ann = torch.cat((v_pool, a_pool), dim=1)

        o = self.O_enc(model_input_ann)
        w = self.W_enc(w)
        theta_o = self.O_proj(o)
        theta_w = self.W_proj(w)

        if self.use_snn_conversion:
            theta_det = theta_o.detach()
            q = float(min(1.0, max(0.5, self.snn_conv_threshold_percentile)))
            thr = torch.quantile(theta_det.abs().reshape(-1), q).detach()
            neg_ratio = (theta_det < 0).float().mean()
            signed = bool(float(neg_ratio.detach()) > 0.1)
            x_quant = self.fake_snn.quantize(theta_det, threshold=thr, signed=signed)
            z_av_snn = self.fake_snn.ste(theta_o, x_quant)
            return z_av_snn, z_av_snn, theta_w

        eval_z = theta_o
        if self.use_teacher_parallel_snn and (
            self._teacher_gate_active or self.teacher_snn_fusion is not None
        ):
            t_out = self._teacher_parallel_forward(
                model_input_ann, theta_o, a_pool, v_pool, v_seq, epoch=None
            )
            teacher_z_snn = t_out["teacher_z_snn"]
            teacher_z_fused = t_out["teacher_z_fused"]
            if teacher_z_fused is not None:
                r = self.teacher_eval_repr
                if r == "ann":
                    eval_z = theta_o
                elif r == "snn":
                    # Diagnostic only (e.g. f_refined - theta_o), not a classification head.
                    eval_z = teacher_z_snn if teacher_z_snn is not None else theta_o
                elif r == "fused":
                    eval_z = teacher_z_fused
                else:
                    eval_z = theta_o

        return eval_z, eval_z, theta_w


def build_clipclap_model(model_params, input_size_audio, input_size_video):
    backend = model_params.get("model_backend", "ann")
    if backend not in ("ann", "snn"):
        raise ValueError(f"Unknown model_backend: {backend!r}, expected 'ann' or 'snn'.")
    return ClipClap_model(model_params, input_size_audio, input_size_video)
