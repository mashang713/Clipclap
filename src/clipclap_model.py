# system, numpy
import logging
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
logger = logging.getLogger(__name__)
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

    Optional AIF (Activation-aware redistribution): single-route dynamics with channel-wise
    threshold and optional membrane offset, injected from offline ANN activation stats.
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
        use_aif=False,
    ):
        super().__init__()
        if snn is None or snn_surrogate is None:
            raise ImportError(
                "SNN backend requires snntorch. Install with: pip install snntorch"
            )
        self.num_steps = int(num_steps)
        self.hidden_size = hidden_size
        self.use_aif = bool(use_aif)

        spike_grad = snn_surrogate.fast_sigmoid()

        # AIF calibration buffers (injected after construction).
        self._aif_mode = "none"
        self.register_buffer("_aif_threshold_1", torch.tensor([]), persistent=False)
        self.register_buffer("_aif_offset_1", torch.tensor([]), persistent=False)
        self.register_buffer("_aif_threshold_2", torch.tensor([]), persistent=False)
        self.register_buffer("_aif_offset_2", torch.tensor([]), persistent=False)

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

        if self.use_aif:
            logger.info("AIF enabled in SNN_EmbeddingNet (calibration will be injected at runtime)")

    def set_aif_params(self, layer_idx: int, threshold_c: torch.Tensor, offset_c: torch.Tensor, mode: str):
        """
        Inject channel-wise AIF calibration into this embedding net.
        layer_idx: 1 for lin1, 2 for lin2 (only if hidden_size>0).
        threshold_c/offset_c are 1D tensors of length C_out for that layer.
        """
        if threshold_c is None:
            threshold_c = torch.tensor([], device=self.lin1.weight.device)
        if offset_c is None:
            offset_c = torch.tensor([], device=self.lin1.weight.device)
        if layer_idx == 1:
            self._aif_threshold_1 = threshold_c.detach()
            self._aif_offset_1 = offset_c.detach()
        elif layer_idx == 2:
            self._aif_threshold_2 = threshold_c.detach()
            self._aif_offset_2 = offset_c.detach()
        else:
            raise ValueError("layer_idx must be 1 or 2")
        self._aif_mode = str(mode)

    def _aif_spike(self, mem, cur, threshold_c, offset_c):
        """
        AIF neuron dynamics (channel-wise threshold/offset) with surrogate gradient:
            mem = beta*mem + cur - offset
            spk = H(mem-threshold)
            mem = mem - spk*threshold
        """
        beta = self.lif_kwargs["beta"]
        spike_grad = self.lif_kwargs["spike_grad"]

        if offset_c is None or offset_c.numel() == 0:
            offset = 0.0
        else:
            offset = offset_c.view(1, -1).to(device=cur.device, dtype=cur.dtype)

        if threshold_c is None or threshold_c.numel() == 0:
            thr = self.lif_kwargs["threshold"]
            threshold = torch.as_tensor(thr, device=cur.device, dtype=cur.dtype)
        else:
            threshold = threshold_c.view(1, -1).to(device=cur.device, dtype=cur.dtype)

        mem = beta * mem + cur - offset
        hard = (mem >= threshold).to(dtype=cur.dtype)
        soft = spike_grad(mem - threshold)
        spk = hard + (soft - soft.detach())
        mem = mem - hard * threshold
        return spk, mem

    def forward(self, x):
        if self.hidden_size > 0:
            if self.use_aif:
                return self._forward_two_layer_aif(x)
            return self._forward_two_layer(x)
        if self.use_aif:
            return self._forward_one_layer_aif(x)
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

    def _forward_one_layer_aif(self, x):
        mem = torch.zeros_like(self.lin1(x))
        spike_sum = torch.zeros_like(mem)
        for _ in range(self.num_steps):
            cur = self.lin1(x)
            spk, mem = self._aif_spike(mem, cur, self._aif_threshold_1, self._aif_offset_1)
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

    def _forward_two_layer_aif(self, x):
        mem1 = torch.zeros_like(self.lin1(x))
        z = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(self.lin2(z))
        spike_sum = torch.zeros_like(mem2)
        for _ in range(self.num_steps):
            cur1 = self.lin1(x)
            spk1, mem1 = self._aif_spike(mem1, cur1, self._aif_threshold_1, self._aif_offset_1)
            spk1 = self.dropout1(spk1)
            cur2 = self.lin2(spk1)
            spk2, mem2 = self._aif_spike(mem2, cur2, self._aif_threshold_2, self._aif_offset_2)
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


def _embeddingnet_get_linear_modules(emb: EmbeddingNet):
    """Return (lin1, lin2_or_None) from EmbeddingNet.fc layout."""
    linears = [m for m in emb.fc.children() if isinstance(m, nn.Linear)]
    if len(linears) == 1:
        return linears[0], None
    if len(linears) == 2:
        return linears[0], linears[1]
    raise ValueError(f"Unexpected number of Linear layers in EmbeddingNet: {len(linears)}")


@torch.no_grad()
def collect_ann_activation_stats(
    ann_model: "ClipClap_model",
    data_loader,
    device,
    num_batches: int = 50,
):
    """
    Collect per-channel mean/std of selected ANN Linear outputs.
    Returns dict: { layer_name: {mu: Tensor[C], sigma: Tensor[C], n: int} }
    """
    ann_model.eval()
    ann_model.to(device)

    targets = {}
    o1, _ = _embeddingnet_get_linear_modules(ann_model.O_enc)
    w1, _ = _embeddingnet_get_linear_modules(ann_model.W_enc)
    targets["O_enc.lin1"] = o1
    targets["W_enc.lin1"] = w1

    op1, op2 = _embeddingnet_get_linear_modules(ann_model.O_proj)
    do1, do2 = _embeddingnet_get_linear_modules(ann_model.D_o)
    targets["O_proj.lin1"] = op1
    if op2 is not None:
        targets["O_proj.lin2"] = op2
    targets["D_o.lin1"] = do1
    if do2 is not None:
        targets["D_o.lin2"] = do2

    wp1, _ = _embeddingnet_get_linear_modules(ann_model.W_proj)
    dw1, _ = _embeddingnet_get_linear_modules(ann_model.D_w)
    targets["W_proj.lin1"] = wp1
    targets["D_w.lin1"] = dw1

    stats = {}
    for k, lin in targets.items():
        c = lin.out_features
        stats[k] = {
            "sum": torch.zeros(c, device=device, dtype=torch.float64),
            "sumsq": torch.zeros(c, device=device, dtype=torch.float64),
            "n": 0,
        }

    hooks = []

    def _make_hook(name):
        def hook(_mod, _inp, out):
            y = out.detach()
            if y.dim() != 2:
                y = y.view(y.size(0), -1)
            s = stats[name]
            s["sum"] += y.to(torch.float64).sum(dim=0)
            s["sumsq"] += (y.to(torch.float64) ** 2).sum(dim=0)
            s["n"] += int(y.size(0))
        return hook

    for name, lin in targets.items():
        hooks.append(lin.register_forward_hook(_make_hook(name)))

    for b_idx, (data, _target) in enumerate(data_loader):
        if b_idx >= int(num_batches):
            break
        p = data["positive"]
        x_a = p["audio"].to(device)
        x_v = p["video"].to(device)
        x_t = p["text"].to(device)
        masks = {"audio": p["audio_mask"], "video": p["video_mask"]}
        timesteps = {"audio": p["timestep"]["audio"], "video": p["timestep"]["video"]}
        _ = ann_model.forward(x_a, x_v, x_t, masks, timesteps)

    for h in hooks:
        h.remove()

    out = {}
    for name, s in stats.items():
        n = max(1, s["n"])
        mu = (s["sum"] / n).to(torch.float32).cpu()
        var = (s["sumsq"] / n) - (s["sum"] / n) ** 2
        var = torch.clamp(var, min=0.0)
        sigma = torch.sqrt(var).to(torch.float32).cpu()
        out[name] = {"mu": mu, "sigma": sigma, "n": int(s["n"])}
    return out


def apply_aif_calibration_to_snn(
    snn_model: "ClipClap_model",
    stats: dict,
    mode: str,
    k: float = 3.0,
):
    """
    Apply AIF-lite/full calibration to SNN_EmbeddingNet modules in ClipClap_model.
    mode: 'cw_threshold' or 'cw_threshold_offset'
    """
    if mode not in ("cw_threshold", "cw_threshold_offset"):
        raise ValueError(f"Unknown snn_aif_mode: {mode}")
    logger.info("AIF enabled, mode=%s, k=%s", mode, k)

    def _make(stat_key):
        st = stats.get(stat_key)
        if st is None:
            return None, None, False
        sigma = st["sigma"]
        mu = st["mu"]
        thr = float(k) * sigma
        off = torch.zeros_like(mu) if mode == "cw_threshold" else mu
        return thr, off, True

    mapping = [
        ("O_enc", 1, "O_enc.lin1"),
        ("W_enc", 1, "W_enc.lin1"),
        ("O_proj", 1, "O_proj.lin1"),
        ("O_proj", 2, "O_proj.lin2"),
        ("D_o", 1, "D_o.lin1"),
        ("D_o", 2, "D_o.lin2"),
        ("W_proj", 1, "W_proj.lin1"),
        ("D_w", 1, "D_w.lin1"),
    ]

    for mod_name, layer_idx, stat_key in mapping:
        mod = getattr(snn_model, mod_name, None)
        if mod is None or not isinstance(mod, SNN_EmbeddingNet):
            continue
        thr, off, ok = _make(stat_key)
        if not ok:
            logger.info("AIF stats missing for %s (fallback to global threshold)", stat_key)
            continue
        thr_t = thr.to(mod.lin1.weight.device)
        off_t = off.to(mod.lin1.weight.device)
        mod.set_aif_params(layer_idx=layer_idx, threshold_c=thr_t, offset_c=off_t, mode=mode)
        logger.info(
            "AIF layer=%s thr(mean/min/max)=%.4f/%.4f/%.4f off(mean/min/max)=%.4f/%.4f/%.4f",
            stat_key,
            float(thr.mean()),
            float(thr.min()),
            float(thr.max()),
            float(off.mean()),
            float(off.min()),
            float(off.max()),
        )













































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
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        elif optimizer == 'adam-sam':
            self.optimizer_gen = SAM(self.parameters(), optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                # lr scheduling on base optimizer
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen.base_optimizer, 'max', patience=3, verbose=True)
        else:
            raise NotImplementedError

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_cyc = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        print('Done')

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


    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy):

        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']


        device = theta_w.device

        if self.cross_entropy_loss==True:
            if self.modality == 'audio':
                embeddings_crossentropy = embeddings_crossentropy[:,512:]
            elif self.modality == 'video':
                embeddings_crossentropy = embeddings_crossentropy[:,:512]
            else:
                if self.word_embeddings == 'wavcaps':
                    embeddings_crossentropy = embeddings_crossentropy[:,512:]
                elif self.word_embeddings == 'clip':
                    embeddings_crossentropy = embeddings_crossentropy[:,:512]

            embedding_cross_entropy=self.W_proj(self.W_enc(embeddings_crossentropy))
            Cross_loss=nn.CrossEntropyLoss()
            scores=torch.matmul(theta_o, embedding_cross_entropy.t()) # (bs, 64) x (K_seen, 64).T = (bs, 64) x (64, K_seen) = (bs, K_seen)
            # gt_cross_entropy = [1, 3, 2, 55, 97, 45, ...] list of gt class labels -> shape (bs,)
            l_ce=Cross_loss(scores, gt_cross_entropy)
        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(theta_o, theta_w)
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


        loss_total = l_rec+l_reg+l_ce
        loss_dict = {
            "Loss/total_loss": loss_total.detach().cpu(),
            "Loss/loss_reg": l_reg.detach().cpu(),
            "Loss/loss_cmd_rec": l_rec.detach().cpu(),
            "Loss/cross_entropy": l_ce.detach().cpu()

        }
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(outputs_second, embedding_crossentropy,  cls_numeric)
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

        return theta_o, theta_o, theta_w


def build_clipclap_model(model_params, input_size_audio, input_size_video):
    backend = model_params.get("model_backend", "ann")
    if backend not in ("ann", "snn"):
        raise ValueError(f"Unknown model_backend: {backend!r}, expected 'ann' or 'snn'.")
    return ClipClap_model(model_params, input_size_audio, input_size_video)
