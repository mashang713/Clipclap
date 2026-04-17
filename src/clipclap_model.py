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

    def forward(self, x, spike_diag=None):
        if self.hidden_size > 0:
            return self._forward_two_layer(x, spike_diag)
        return self._forward_one_layer(x, spike_diag)

    def _forward_one_layer(self, x, spike_diag=None):
        mem = torch.zeros_like(self.lin1(x))
        spike_sum = torch.zeros_like(mem)
        for _ in range(self.num_steps):
            cur = self.lin1(x)
            spk, mem = self.lif1(cur, mem)
            spk = self.dropout1(spk)
            spike_sum = spike_sum + spk
        out = spike_sum / self.num_steps
        if spike_diag is not None:
            spike_diag["lif1_mean_firing_rate"] = float(out.mean().detach())
            spike_diag["lif1_nonzero_frac"] = float((out.detach().abs() > 1e-3).float().mean())
            spike_diag["lif1_output_mean"] = float(out.mean().detach())
            spike_diag["lif1_output_std"] = float(out.std().detach())
            spike_diag["lif1_mem_mean_last"] = float(mem.detach().mean())
        return out

    def _forward_two_layer(self, x, spike_diag=None):
        mem1 = torch.zeros_like(self.lin1(x))
        z = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(self.lin2(z))
        spike_sum = torch.zeros_like(mem2)
        spk1_acc = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        for _ in range(self.num_steps):
            cur1 = self.lin1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            spk1_acc = spk1_acc + spk1
            cur2 = self.lin2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            spike_sum = spike_sum + spk2
        out = spike_sum / self.num_steps
        if spike_diag is not None:
            t = float(self.num_steps)
            r1 = spk1_acc / t
            spike_diag["input_mean"] = float(x.mean().detach())
            spike_diag["input_std"] = float(x.std().detach())
            spike_diag["lif1_mean_firing_rate"] = float(r1.mean().detach())
            spike_diag["lif1_nonzero_frac"] = float((r1.detach().abs() > 1e-3).float().mean())
            spike_diag["lif2_mean_firing_rate"] = float(out.mean().detach())
            spike_diag["lif2_nonzero_frac"] = float((out.detach().abs() > 1e-3).float().mean())
            spike_diag["lif2_output_mean"] = float(out.mean().detach())
            spike_diag["lif2_output_std"] = float(out.std().detach())
            spike_diag["lif2_mem_mean_last"] = float(mem2.detach().mean())
        return out

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
            "Loss/cross_entropy": l_ce.detach().cpu(),
        }
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(
                outputs, embedding_crossentropy, cls_numeric
            )

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(
                outputs, embedding_crossentropy, cls_numeric
            )

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(
                    outputs_second, embedding_crossentropy, cls_numeric
                )
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
