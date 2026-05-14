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

        # Teacher parallel: full ANN route (O_enc/O_proj) plus optional SNN route on the same
        # forward inputs a/v. Full SNN route (when enabled) uses trainable blocks only on
        # pre-extracted pooled AV features — not raw waveform / video backbone SNN.
        self.use_teacher_parallel_snn = bool(params_model.get("use_teacher_parallel_snn", False))
        self.teacher_snn_arch = str(params_model.get("teacher_snn_arch", "backend_only")).lower()
        if self.teacher_snn_arch not in ("backend_only", "full_snn_route"):
            raise ValueError(
                f"teacher_snn_arch must be 'backend_only' or 'full_snn_route', got {self.teacher_snn_arch!r}"
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
        if self.teacher_snn_fusion_mode not in ("add", "gated_scale"):
            raise ValueError(
                f"teacher_snn_fusion_mode must be 'add' or 'gated_scale', got {self.teacher_snn_fusion_mode!r}"
            )
        self.teacher_ann_gate_snn = bool(params_model.get("teacher_ann_gate_snn", False))
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
        self.teacher_snn_fusion = None
        self.teacher_snn_input_size = None
        self.teacher_audio_snn_front = None
        self.teacher_video_snn_front = None
        if self.use_teacher_parallel_snn:
            if self.modality == "both":
                fusion_in = 1536
            elif self.modality == "audio":
                fusion_in = 1024
            else:
                fusion_in = 512
            self.teacher_snn_input_size = int(fusion_in)
            out_dim = int(self.dim_out)
            hid = int(params_model.get("teacher_snn_hidden_dim", 512))
            self.teacher_snn_fusion = TeacherSNNFusionBranch(
                input_size=self.teacher_snn_input_size,
                hidden_size=hid,
                output_size=out_dim,
                num_steps=int(params_model.get("teacher_snn_timesteps", 4)),
                beta=float(params_model.get("teacher_snn_decay", 0.9)),
                threshold=float(params_model.get("teacher_snn_threshold", 1.0)),
                dropout=float(params_model.get("teacher_snn_dropout", 0.1)),
                ann_dim=out_dim,
                use_ann_gate=self.teacher_ann_gate_snn,
                gate_strength=self.teacher_gate_strength,
                leak_strength=self.teacher_leak_strength,
            )
            if self.teacher_snn_arch == "full_snn_route":
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
        self.debug_print_shapes = bool(params_model.get("debug_print_shapes", False))
        self._forward_shape_debug_entry_printed = False
        self._forward_shape_debug_printed = False

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def forward(self, a, v, w, masks, timesteps):
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
            _shape_desc("v (raw forward input)", v, pre_lines)
            _shape_desc("w (forward arg cls_embedding, raw)", w_cls_raw, pre_lines)
            _shape_desc("masks", masks, pre_lines)
            _shape_desc("timesteps", timesteps, pre_lines)
            print("\n".join(pre_lines), flush=True)

        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)
        if self.modality == 'audio':
            w = w[:,512:]
            model_input_ann = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input_ann = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input_ann = torch.cat((v, a), dim=1)

        model_input = model_input_ann

        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)


        rho_o = self.D_o(theta_o)


        theta_w = self.W_proj(w)

        teacher_z_snn = None
        teacher_z_fused = None
        teacher_theta_scaled = None
        teacher_spike_count = None
        teacher_spike_rate = None
        teacher_spike_scale = None
        teacher_gate_mean = None
        teacher_fire_rate_mean = None
        teacher_fusion_mode = None
        teacher_model_input_ann = None
        teacher_model_input_snn = None
        teacher_theta_ann = None
        teacher_audio_snn_front = None
        teacher_video_snn_front = None
        teacher_audio_front_spike_count = None
        teacher_audio_front_spike_rate = None
        teacher_audio_front_spike_scale = None
        teacher_audio_front_fire_rate_mean = None
        teacher_video_front_spike_count = None
        teacher_video_front_spike_rate = None
        teacher_video_front_spike_scale = None
        teacher_video_front_fire_rate_mean = None

        if self.use_teacher_parallel_snn and self.teacher_snn_fusion is not None:
            teacher_model_input_ann = model_input_ann
            teacher_theta_ann = theta_o

            aux_a = None
            aux_v = None
            if self.teacher_snn_arch == "full_snn_route":
                if self.modality == "both":
                    a_snn, aux_a = self.teacher_audio_snn_front(a)
                    v_snn, aux_v = self.teacher_video_snn_front(v)
                    teacher_audio_snn_front = a_snn
                    teacher_video_snn_front = v_snn
                    model_input_snn = torch.cat((v_snn, a_snn), dim=1)
                elif self.modality == "audio":
                    a_snn, aux_a = self.teacher_audio_snn_front(a)
                    teacher_audio_snn_front = a_snn
                    model_input_snn = a_snn
                else:
                    v_snn, aux_v = self.teacher_video_snn_front(v)
                    teacher_video_snn_front = v_snn
                    model_input_snn = v_snn

                if aux_a is not None:
                    teacher_audio_front_spike_count = aux_a["spike_count"]
                    teacher_audio_front_spike_rate = aux_a["spike_rate"]
                    teacher_audio_front_spike_scale = aux_a["spike_scale"]
                    teacher_audio_front_fire_rate_mean = aux_a["fire_rate_mean"]
                if aux_v is not None:
                    teacher_video_front_spike_count = aux_v["spike_count"]
                    teacher_video_front_spike_rate = aux_v["spike_rate"]
                    teacher_video_front_spike_scale = aux_v["spike_scale"]
                    teacher_video_front_fire_rate_mean = aux_v["fire_rate_mean"]
            else:
                model_input_snn = model_input_ann

            teacher_model_input_snn = model_input_snn

            assert model_input_snn.shape[1] == self.teacher_snn_input_size, (
                f"Teacher SNN backend input dim mismatch: "
                f"model_input_snn has {model_input_snn.shape[1]}, "
                f"backend expects {self.teacher_snn_input_size}"
            )
            ann_ctx = theta_o if self.teacher_ann_gate_snn else None
            teacher_z_snn, teacher_aux = self.teacher_snn_fusion(
                model_input_snn, ann_context=ann_ctx
            )
            teacher_spike_count = teacher_aux["spike_count"]
            teacher_spike_rate = teacher_aux["spike_rate"]
            teacher_spike_scale = teacher_aux["spike_scale"]
            teacher_gate_mean = teacher_aux["gate_mean"]
            teacher_fire_rate_mean = teacher_aux["fire_rate_mean"]
            teacher_fusion_mode = self.teacher_snn_fusion_mode
            if self.teacher_snn_fusion_mode == "gated_scale":
                teacher_theta_scaled = theta_o * (
                    1.0
                    + self.teacher_spike_scale_strength
                    * (2.0 * teacher_spike_scale - 1.0)
                )
                teacher_z_fused = (
                    teacher_theta_scaled + self.teacher_snn_gamma * teacher_z_snn
                )
            else:
                teacher_theta_scaled = None
                teacher_z_fused = theta_o + self.teacher_snn_gamma * teacher_z_snn

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
                if self.teacher_snn_arch == "full_snn_route":
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
                self.use_teacher_parallel_snn and self.teacher_snn_fusion is not None
            ),
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

        teacher_z_snn = outputs.get("teacher_z_snn")
        teacher_z_fused = outputs.get("teacher_z_fused")

        device = theta_w.device
        loss_teacher_fire = None
        loss_teacher_front_fire = None

        l_ann_teacher = torch.tensor(0.0, device=device)
        l_snn_teacher = torch.tensor(0.0, device=device)
        l_fused_teacher = torch.tensor(0.0, device=device)
        teacher_parallel_ce_used = False

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
            use_teacher_parallel_ce = (
                self.use_teacher_parallel_snn
                and outputs.get("teacher_parallel_enabled")
                and teacher_z_snn is not None
                and teacher_z_fused is not None
                and embedding_cross_entropy is not None
            )
            if embedding_cross_entropy is None:
                l_ce = torch.tensor(0., device=device)
            elif use_teacher_parallel_ce:
                teacher_parallel_ce_used = True

                def _teacher_ce_logits(z):
                    return torch.matmul(z, embedding_cross_entropy.t())

                l_ann_teacher = Cross_loss(_teacher_ce_logits(theta_o), gt_cross_entropy)
                l_snn_teacher = Cross_loss(_teacher_ce_logits(teacher_z_snn), gt_cross_entropy)
                l_fused_teacher = Cross_loss(_teacher_ce_logits(teacher_z_fused), gt_cross_entropy)
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
            and self.teacher_fire_rate_reg > 0.0
            and outputs.get("teacher_spike_rate") is not None
        ):
            sr = outputs["teacher_spike_rate"]
            loss_teacher_fire = ((sr - self.teacher_fire_rate_target) ** 2).mean()
            loss_original = loss_original + self.teacher_fire_rate_reg * loss_teacher_fire

        if (
            self.use_teacher_parallel_snn
            and self.teacher_snn_arch == "full_snn_route"
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
            loss_dict["Loss/loss_teacher_snn"] = l_snn_teacher.detach().cpu()
            loss_dict["Loss/loss_teacher_fused"] = l_fused_teacher.detach().cpu()
            loss_dict["Loss/loss_teacher_ce"] = l_ce.detach().cpu()
            loss_dict["Diag/teacher_snn_gamma"] = torch.tensor(float(self.teacher_snn_gamma))
            loss_dict["Diag/teacher_snn_alpha"] = torch.tensor(float(self.teacher_snn_alpha))
            loss_dict["Diag/teacher_snn_beta"] = torch.tensor(float(self.teacher_snn_beta))
        if outputs.get("teacher_parallel_enabled") and outputs.get("teacher_spike_rate") is not None:
            loss_dict["Diag/teacher_fire_rate_mean"] = outputs["teacher_fire_rate_mean"].detach().cpu()
            loss_dict["Diag/teacher_gate_mean"] = outputs["teacher_gate_mean"].detach().cpu()
            loss_dict["Diag/teacher_spike_scale_mean"] = outputs["teacher_spike_scale"].mean().detach().cpu()
            loss_dict["Diag/teacher_fusion_mode_gated"] = torch.tensor(
                1.0 if str(outputs.get("teacher_fusion_mode") or "") == "gated_scale" else 0.0
            )
            if loss_teacher_fire is not None:
                loss_dict["Loss/loss_teacher_fire_rate"] = loss_teacher_fire.detach().cpu()
            else:
                loss_dict["Loss/loss_teacher_fire_rate"] = torch.tensor(0.0)
        if outputs.get("teacher_parallel_enabled") and str(
            outputs.get("teacher_snn_arch") or ""
        ) == "full_snn_route":
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
