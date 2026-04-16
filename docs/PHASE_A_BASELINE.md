# Phase A — Canonical SNN-head baseline (frozen)

**Scope:** Trainable path = **SNN** (`SNN_EmbeddingNet` for `O_enc`, `W_enc`, `O_proj`, `D_o`, `W_proj`, `D_w`). **No** geometry-KD, **no** frozen ANN teacher at runtime. **Inputs** remain **offline** CLIP/CLAP (and text) features from `_features_processed/…pkl` — Phase B will replace raw front-ends separately.

---

## Reproducible command (template)

Set paths to your machine. **ANN checkpoint** is only for **optional** `--snn_init_ann_path` (copy fused Linear weights into SNN); use a checkpoint from a run with `model_backend: ann`.

```bash
cd /path/to/ClipClap-GZSL
conda activate clipclap   # or your env

export UCF_ROOT=/path/to/UCF_data_root
export LOG_DIR=/path/to/ClipClap-GZSL/runs
export ANN_CKPT=/path/to/ANN/ClipClap_model_score_ckpt_14.pt   # optional; omit flag for random SNN init

python main.py --cfg config/clipclap_snn_baseline.yaml \
  --device cuda:0 \
  --root_dir "$UCF_ROOT" \
  --log_dir "$LOG_DIR" \
  --dataset_name UCF \
  --snn_init_ann_path "$ANN_CKPT" \
  --run all
```

To omit ANN init, drop `--snn_init_ann_path "$ANN_CKPT"` (SNN trains from default LIF init).

Override training length, e.g. `--epochs 50`, when matching a full baseline (yaml default may be `epochs: 1` for smoke tests).

---

## Module status (Phase A)

| Component | SNN? | Notes |
|-----------|------|--------|
| Video / audio **features** in dataloader | No | Precomputed floats from CLIP / CLAP (or WavCaps) pipelines |
| Class **text** embeddings (`word_embeddings: both`) | No | Precomputed CLIP + WavCaps (or config variant) |
| `O_enc`, `W_enc`, `O_proj`, `D_o`, `W_proj`, `D_w` with `model_backend: snn` | **Yes** | `snntorch` Leaky LIF stacks |
| `model_backend: ann` | No | `EmbeddingNet` (Linear+BN+ReLU) |
| Losses (`l_ce`, `l_reg`, `l_rec`) | N/A | Standard PyTorch |
| GZSL **evaluation** (`get_evaluation` → `test`) | N/A | `cdist` / HM-ZSL on embeddings |
| `--snn_init_ann_path` | N/A | One-time **weight copy** from ANN ckpt, not a runtime teacher |

---

## What was removed for Phase A cleanliness

- Geometry-KD / frozen teacher / related CLI and yaml keys
- Extra checkpoint stripping for `_geometry_teacher` (no longer created)

---

## Next (not Phase A)

- **Phase B:** Replace video/audio **feature extraction** with trainable (e.g. SNN) front-ends; keep this doc as the offline-feature baseline reference.
