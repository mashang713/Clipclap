# Phase B — Minimal audio skeleton (single modality)

## 1. Modality choice: **audio** first

- **Lower IO / memory** than video (no frame decoding stack in v1).
- Natural **1-D / 2-D** front-end (log-mel) maps cleanly to a small **SNN MLP/Conv** before the existing head.
- UCF audio features are **128-D CLAP time series** in Phase A; reshaping to a **fixed `mel_flat_dim` (4096)** is a minimal bridge before real `.wav` loading.

Video Phase B can mirror this later with a separate dataset + frontend.

---

## 2. New files

| File | Role |
|------|------|
| `src/phase_b/__init__.py` | Package exports |
| `src/phase_b/audio_frontend.py` | `PhaseBAudioEncoderSNN`: mel-flat → **1024-D** (matches `O_enc` input for `modality: audio`) |
| `src/phase_b/ucf_phase_b_audio_dataset.py` | `ContrastivePhaseBAudio` / `PhaseBAudioSource`, replaces contrastive audio with `(1, 4096)` mel-flat |
| `src/phase_b/phase_b_audio_model.py` | `ClipClapPhaseB_AudioWrapper`: runs frontend, then existing `ClipClap_model` |
| `src/phase_b/run_phase_b_ucf_audio.py` | UCF-only runner (not wired to `main.py`) |
| `scripts/phase_b_train_audio.py` | CLI entry |
| `config/phase_b_ucf_audio.yaml` | Example config (`modality: audio`, `model_backend: snn`) |

---

## 3. New classes

- **`PhaseBAudioEncoderSNN`**: `SNN_EmbeddingNet(mel_flat_dim → 1024)`.
- **`ClipClapPhaseB_AudioWrapper`**: `optimize_params` / `forward` / `get_embeddings` apply frontend on audio batch, delegate to inner `ClipClap_model`.
- **`ContrastivePhaseBAudio`** (alias `UCFPhaseBAudioDataset`): subclasses `ContrastiveDataset`; overrides `__getitem__` to replace `audio` only.
- **`PhaseBAudioSource`**: `dummy` \| `offline_as_mel` (CLI: `--phase_b_audio_source`).

---

## 4. Data flow

1. `UCFDataset` (unchanged) loads `_features_processed/*.pkl` for **labels + text + offline video/audio** as in Phase A.
2. `ContrastivePhaseBAudio` samples positive/negative pairs like Phase A.
3. **Audio only** is replaced by a **4096-D row** (`(1, 4096)` numpy) — from **flattened offline audio** or **Gaussian dummy**.
4. `DefaultCollator` pads/trims as usual → batch tensor `(B, T, 4096)` (typically `T` set by fixed mode).
5. **Wrapper** mean-pools time if needed → `(B, 4096)` → **frontend** → `(B, 1024)`.
6. Inner **`ClipClap_model`** (`modality: audio`) treats that as `a` → **`O_enc` → `O_proj` / `D_o`**, **`W_enc` → `W_proj` / `D_w`** with **frozen text rows** from pickle (unchanged).

---

## 5. Encoder output dimension

- **Frontend output:** **1024** (required by `O_enc` / `W_enc` for `modality: audio`).
- **Mel-flat width:** **4096** (configurable constant in `PhaseBAudioEncoderSNN` / dataset; keep them aligned).

---

## 6. What stays Phase A

- **Text / class embeddings** from the same UCF processed pickle.
- **Video** tensors in the batch (still loaded; unused for `O_enc` when `modality: audio`, but collator/eval paths remain compatible).
- **Losses** (`l_ce`, `l_reg`, `l_rec`), **metrics**, **GZSL eval** protocol.
- **`main.py`** and default **`python main.py ...`** unchanged.

---

## 7. Explicit non-goals (this PR)

- No video front-end, no dual-modality fusion changes.
- No geometry-KD, no new text encoder.
- No calibrated stacking changes.
- No claim on SOTA metrics — **runnable wiring only**.

---

## 8. Runs (B-0 / B-1 / eval)

### Phase B-0 — pipeline smoke (no signal)

- Config: `config/phase_b_ucf_audio_debug.yaml` (`phase_b_audio_source: dummy`, `n_batches: 15`, `epochs: 1`, `save_checkpoints: true`).
- Goal: data load, frontend → 1024, train/val, checkpoint write, **post-train eval** (if `save_checkpoints`).

```bash
python scripts/phase_b_train_audio.py -c config/phase_b_ucf_audio_debug.yaml \
  --root_dir /path/to/UCF --log_dir /path/to/runs --device cuda:0
```

### Phase B-1 — minimal “real” mel bridge

- Config: `config/phase_b_ucf_audio.yaml` (`offline_as_mel`, `save_checkpoints: true`).
- Optional ANN init for the **head** only: `--snn_init_ann_path /path/to/ANN_clipclap_ckpt.pt`

```bash
python scripts/phase_b_train_audio.py -c config/phase_b_ucf_audio.yaml \
  --root_dir /path/to/UCF --log_dir /path/to/runs --device cuda:0 \
  --snn_init_ann_path /path/to/ANN_ckpt.pt \
  --phase_b_audio_source offline_as_mel
```

After training, **standalone eval** (same Phase B model + loaders):

```bash
python scripts/phase_b_eval_ucf.py --run_dir /path/to/runs/Apr??_... \
  --root_dir /path/to/UCF --device cuda:0
```

### Phase A baseline (对照)

```bash
python main.py --cfg config/clipclap_snn_baseline.yaml --run all \
  --root_dir /path/to/UCF --log_dir /path/to/runs --device cuda:0 \
  --modality both \
  --snn_init_ann_path /path/to/ANN_ckpt.pt
```

Use **`--modality audio`** for audio-only Phase A vs Phase B (fairer modality match).

### What to log for the comparison table

| Run | Seen | Unseen | HM | ZSL | Notes |
|-----|------|--------|-----|-----|-------|
| Phase A | | | | | train loss / NaN |
| Phase B | | | | | |

Use `--phase_b_audio_source dummy` only for B-0 wiring checks; B-1 should use `offline_as_mel`.
