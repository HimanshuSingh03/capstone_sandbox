Look Once To Hear (Target-Speaker Hearing)

This repository implements a fast, streaming, speaker‑conditioned binaural speech extraction system. It takes a noisy multi‑talker binaural mixture and an enrollment of the target speaker, and returns a target‑enhanced binaural signal that preserves spatial cues (ILD/ITD). Training and evaluation are built around a synthetic, reproducible pipeline using LibriSpeech content rendered through measured/simulated HRTFs.

If you’re familiar with target‑speaker separation, this repo combines:
- A streaming TF‑GridNet separator conditioned on a target speaker embedding.
- A dataset generator that mixes speech/noise and renders them binaurally with diverse HRTFs (with optional source motion and augmentation).
- Evaluation that measures separation quality (SI‑SNRi) and spatial cue fidelity (ITD/ILD errors).

Paper reference: Look Once To Hear (link: https://arxiv.org/pdf/2405.06289v3). The implementation here follows that overall problem and design: use one enrollment “look” to get an embedding that conditions a streaming separator, optimized for real‑time target hearing.


At A Glance

- Problem: Hear your person of interest in a noisy, multi‑talker scene, preserving spatial cues.
- Inputs: binaural mixture [B, 2, T] + target enrollment utterance → target embedding [B, 1, E].
- Model: streaming TF‑GridNet with speaker conditioning and local attention.
- Output: enhanced target binaural audio [B, 2, T], suitable for low‑latency playback.
- Data: synthetic binaural mixes from LibriSpeech using diverse HRTFs; optional moving sources and noisy enrollments for robustness.


Quick Start

- Environment: follow SETUP_GUIDE.md for dependencies. The main libraries include PyTorch, PyTorch Lightning, torchaudio, asteroid‑filterbanks, scaper, wandb.
- Configs: see configs/tsh.json for target‑hearing training; configs/embed.json for the optional binaural embedding pretraining.
- Train target‑hearing model:
  - Example: `python src/trainer.py --config configs/tsh.json --run_dir runs/tsh`
- Validate/test: use `--test` with the same command and optionally `--ckpt` to point to a checkpoint.
- Evaluate with CSV output: `python src/ts_hear_test.py --device 0` (edit the config paths near the bottom of the file or pass your own via CLI by modifying the script).


Two‑Minute Intro (Recipes)

- Train Target‑Hearing
  - `python src/trainer.py --config configs/tsh.json --run_dir runs/tsh`
  - Checkpoints: `runs/tsh/last.ckpt` and `runs/tsh/best/*.ckpt`

- Test With Best Checkpoint
  - `python src/trainer.py --config configs/tsh.json --run_dir runs/tsh --test`

- Learn Binaural Embeddings (optional)
  - `python src/trainer.py --config configs/embed.json --run_dir runs/embed`

- Compute d‑Vectors From LibriSpeech‑style Folders
  - `python src/datasets/dvector_embeddings.py --root_dir <scaper_fmt> --output_dir <emb_out>`

- Minimal Demo (interactive)
  - See `my_demo/interactive_demo.py` for a simple in/out audio demo you can adapt.


Minimal End‑to‑End Example (Code)

This shows how to load a trained target‑hearing model and run one forward pass with a mixture and a precomputed embedding.

```python
import torch
from src import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config and model
hparams = utils.Params('configs/tsh.json')
pl_module = utils.import_attr(hparams.pl_module)(**hparams.pl_module_args)
state = torch.load('runs/tsh/best/<your_ckpt>.ckpt', map_location='cpu')
pl_module.load_state_dict(state['state_dict'])
pl_module.eval().to(device)

# Prepare inputs
# mixture: [B, 2, T]; embedding_gt: [B, 1, E]
mixture = torch.randn(1, 2, 16000 * 5, device=device)
embedding = torch.randn(1, 1, 256, device=device)

with torch.no_grad():
    enhanced = pl_module.model(mixture, embedding)  # [B, 2, T]
```

Tip: For real enrollment audio, you can either use the learned binaural embedding model (configs/embed.json), or precompute d‑vectors with `src/datasets/dvector_embeddings.py` and load the target’s vector.


How The System Works

High‑Level Flow

- Input mixture: a multi‑speaker binaural signal (2 channels) plus background noise.
- Enrollment: a short utterance from the target speaker (binaural and/or anechoic), from which a fixed‑dimensional embedding is derived.
- Separator: a streaming TF‑GridNet variant conditioned on the embedding; it estimates the target’s binaural signal.
- Training: pairs mixtures with the target reference; optimizes SNR/SI‑SNR based losses; logs metrics and, optionally, embeddings.
- Inference: compute or load the target embedding; run the separator in a streaming fashion; report SI‑SNRi and spatial cue errors (ITD/ILD).


Audio, Features, and Streaming

- Binaural rendering. Datasets are synthesized by convolving mono sources with HRTFs/BRIRs to produce [2, T] signals per source and noise, then mixing and normalizing. Multiple HRTF collections are supported to improve robustness (CIPIC, RRBRIR, ASH, CATT, PRA). See src/datasets/multi_ch_simulator.py.
- Optional motion. Sources can move via a motion simulator that updates source direction per frame, then applies time‑varying HRTFs. See src/datasets/motion_simulator.py (CIPICMotionSimulator2).
- TF front‑end and streaming. The separator operates on chunked STFT frames with overlap‑add and keeps internal state across chunks for causal or chunk‑causal streaming:
  - Encoder/decoder come from asteroid‑filterbanks (STFT)
  - Internal buffers for conv, deconv, and ISTFT overlap history are maintained between chunks to enable low‑latency inference. See src/models/tfgridnet_realtime/tfgridnet_causal.py.
- Speaker conditioning. A target speaker embedding (e.g., d‑vector) is projected and fused with the TF representation to bias the separator toward the target voice. See embed_to_feats_proj and block modulation in src/models/tfgridnet_realtime/tfgridnet_causal.py.


Models

- Streaming TF‑GridNet (realtime):
  - File: src/models/tfgridnet_realtime/net.py
  - Core: src/models/tfgridnet_realtime/tfgridnet_causal.py
  - Key parameters: `stft_chunk_size`, `stft_pad_size` (lookahead), number of layers/heads, use of local attention, and `chunk_causal` mode.
  - Conditioning: `spk_emb_dim` projects the enrollment embedding into TF feature space to modulate blocks.
  - Output: target binaural waveform aligned with input length (with padding trimmed), suitable for continuous streaming via `init_buffers`/`predict`.

- Offline TF‑GridNet (original):
  - File: src/models/tfgridnet_orig/tfgridnet.py
  - Provides a standard non‑streaming reference; also includes an embedding extractor variant `EmbedTFGridNet` used for binaural embedding learning.


Embeddings and Enrollment

- d‑vectors from enrollment audio:
  - During dataset creation, per‑utterance embeddings are precomputed and cached (see src/datasets/dvector_embeddings.py using Resemblyzer; an alternative NEMO/TiTANet pipeline is in src/datasets/nemo_embed.py).
  - Training uses these to supply ground‑truth target embeddings (`embedding_gt`) and negative samples (`embedding_neg`).
- Optional binaural embedding model:
  - You can train a 2‑ch embedding network to map binaural enrollments to the same space as the cached d‑vectors using cosine embedding loss with margin and hard negatives.
  - See src/binaural_embed_pl_module.py with model config in configs/embed.json.
- Inference options (src/ts_hear_test.py):
  - Use ground‑truth cached embeddings from the dataset.
  - Use a trained `EmbedTFGridNet` on the binaural enrollment to produce an embedding.
  - Or compute a d‑vector from the (denoised) enrollment audio via Resemblyzer (`--embed_from_wav`).


Dataset Layout (Example)

```text
data/
  MixLibriSpeech/
    librispeech_scaper_fmt/
      train-clean-360/<speaker_id>/<utterance>.wav
      dev-clean/...
      test-clean/...
    jams/
      train-clean-360/<mix_id>/mixture.jams
      dev-clean/...  test-clean/...
    librispeech_dvector_embeddings/
      train-clean-360/<speaker_id>.pt   # dict: filename -> d‑vector
      dev-clean/...  test-clean/...
    CIPIC/train_hrtf.txt  # list of SOFA files
    CIPIC/val_hrtf.txt
    CIPIC/test_hrtf.txt
  RRBRIR/...
  ASH-Listening-Set-8.0/BRIRs/...
  CATT_RIRs/Binaural/16k/...
```


Training

- Entry point: src/trainer.py (PyTorch Lightning)
  - Loads `pl_module`, datasets, callbacks, logging; runs DDP if multiple GPUs.
  - Checkpointing: saves `last.ckpt` and top‑K best to `runs/<name>/best/` by a monitored metric.
  - Resumes from `--ckpt` or from `last.ckpt` in the run directory.

- Target‑Hearing PLModule: src/ts_hear_embed_pl_module.py
  - Forward: `model(mixture, embedding)` where `mixture` is [B, C=2, T] and `embedding` is [B, 1, E].
  - Loss: default is negative SNR (`-SNR`) between estimate and target; SI‑SNRi and SNRi are logged against the original mixture as improvement metrics.
  - Optional direction regularization: cross‑entropy on a discretized interaural time difference proxy from `tgt_shift` (not enabled by default configs).
  - Scheduler: ReduceLROnPlateau supported via config.
  - Logging: embeddings and metrics logged to W&B via src/ts_hear_embed_pl_module.Logger.

- Binaural Embedding PLModule: src/binaural_embed_pl_module.py
  - Model: `EmbedTFGridNet` on 2‑ch enrollment audio.
  - Loss: cosine embedding loss with positive pair (enrollment vs `embedding_gt`) and, after a warm‑up epoch, negatives from `embedding_neg`.
  - Monitors `val/loss` (lower is better).


Common Pitfalls

- Paths invalid in configs: make sure `fg_dir`, `bg_dir`, `jams_dir`, `embed_dir`, and all `hrtf_list` entries exist.
- W&B login: if you don’t want to log online, set `WANDB_DISABLED=true` or `WANDB_MODE=offline` before running.
- GPU memory: reduce `batch_size` or STFT sizes (`stft_chunk_size`, layers/heads) if you hit OOM.
- Noisy enrollments: ensure `num_enroll=1` in the noisy‑enroll datasets (as expected by the code).
- Motion: if HRTF motion libs aren’t compiled, disable motion (`use_motion: false`) or switch `hrtf_type` to a static backend.


Datasets and Synthesis

- Base LibriSpeech + Scaper format:
  - JAMS annotations drive dynamic mixing: backgrounds + 2–3 foreground talkers with SNR and duration control.
  - See src/datasets/MixLibriSpeech*.py and the generation utilities in src/datasets/generate_jams.py and src/datasets/librispeech2scaper.py.

- Binaural rendering backends (src/datasets/multi_ch_simulator.py):
  - CIPIC (subject‑specific SOFA), RRBRIR, ASH, CATT, PRA; `MultiChSimulator` mixes across sets with configured sampling weights to improve generalization.
  - Output: per‑source binaural signals and a binaural noise track; mixed and peak‑normalized.

- Motion (src/datasets/motion_simulator.py):
  - CIPICMotionSimulator2 renders moving sources using a C++ core via ctypes; provides either smooth angular velocity or piecewise arcs.
  - The dataset can export target angular velocity and enrollment direction error for analysis (`tgt_ang_vel`, `tgt_enroll_error`).

- Noisy enrollments and augmentation:
  - The `MixLibriSpeechNoisyEnroll*` variants create a separate pseudo‑scene for the enrollment utterance, rendered binaurally, then add noise (scalable, plus optional white/pink/brown noise) and normalize.
  - Ground‑truth d‑vectors are taken from the clean anechoic enrollment file to keep the embedding target stable.

- Provided dataset classes (major ones):
  - src/datasets/MixLibriSpeech.py: static JAMS‑driven mixing + binaural rendering.
  - src/datasets/MixLibriSpeechDM.py: dynamic mixing via Scaper at training time.
  - src/datasets/MixLibriSpeechNoisyEnroll.py and src/datasets/MixLibriSpeechNoisyEnrollNorm.py: noisy enrollment pipelines (the latter includes additional stats like angular velocity and anechoic enrollment for analysis).
  - src/datasets/MixLibriSpeechNoisyEnrollDirectional.py, src/datasets/MixLibriSpeechNoisyEnrollWP.py, src/datasets/MixLibriSpeechMotion.py: alternative variants for ablations (direction supervision, different noise policies, motion).


Evaluation and Metrics

- Separation metrics: SI‑SNR and SI‑SNRi (improvement vs input mixture). Logged during train/val/test (see src/ts_hear_embed_pl_module.py) and exported in src/ts_hear_test.py.
- Spatial cue fidelity (src/eval/binaural.py):
  - ILD error: difference in interaural level difference between estimate and target.
  - ITD error: difference in interaural time delay via cross‑correlation; supports moving‑source chunking.
- Test script: src/ts_hear_test.py
  - Loads the trained separator and optional embedding model.
  - Computes embeddings as configured, runs inference, assembles a per‑sample DataFrame with file‑level context (source files, genders, input similarity), and writes a CSV.


FAQ

- Do I need an embedding model? No. You can use cached d‑vectors (fastest path). The embedding model helps if you prefer end‑to‑end binaural enrollment.
- How do I change HRTF backends? In the config, set `hrtf_type` to one of `CIPIC`, `RRBRIR`, `ASH`, `CATTRIR`, `PRA`, or `MultiCh` for a mix. Update `hrtf_list` paths accordingly.
- Can I run low‑latency streaming? Yes. Use `Net.init_buffers()` and `Net.predict()` in `src/models/tfgridnet_realtime/net.py` to feed chunked audio in real time.
- What loss should I use? The default `-SNR` is simple and strong. For scale invariance or perceptual bias, switch to SI‑SDR or mel/CDPAM in `src/losses/LossFn.py` and update the PLModule.


Configuration Files

- configs/tsh.json
  - Uses `src.ts_hear_embed_pl_module.PLModule`.
  - Model: `src.models.tfgridnet_realtime.net.Net` with causal streaming, attention, and lookahead.
  - Dataset: `src.datasets.MixLibriSpeechNoisyEnrollNorm.MixLibriSpeechNoisyEnroll` for train/val/test, multi‑backend HRTFs (`hrtf_type: MultiCh`).
  - Batch sizes, LR, scheduler, and data paths are specified here.

- configs/embed.json
  - Uses `src.binaural_embed_pl_module.PLModule` with `EmbedTFGridNet` to learn a binaural enrollment encoder to the d‑vector space.


File‑by‑File Guide

- Top level
  - README.md: this document.
  - SETUP_GUIDE.md: environment setup instructions and dependencies.
  - requirements.txt: Python packages list.
  - slurm.py: convenience wrapper for SLURM job submission (cluster training).
  - LICENSE: project license.
  - test_convert.py: small utility/test script for local conversions.

- Training and orchestration
  - src/trainer.py: Lightning training entry; loads configs, datasets, PLModule, logging, checkpoints.
  - src/utils.py: utilities for dynamic imports (`import_attr`), JSON params loading/saving (`Params`), W&B run IDs.

- PyTorch Lightning modules
  - src/ts_hear_embed_pl_module.py: target‑hearing PLModule; training/validation/test steps; logs SI‑SNRi, optional direction loss; LR scheduler plumbing.
  - src/binaural_embed_pl_module.py: embedding PLModule using cosine embedding loss with negatives.

- Models
  - src/models/tfgridnet_realtime/net.py: wrapper with STFT chunking and stateful streaming (`init_buffers`, `predict`).
  - src/models/tfgridnet_realtime/tfgridnet_causal.py: causal TF‑GridNet core with local attention and speaker conditioning.
  - src/models/tfgridnet_realtime/film.py: FiLM‑style modulation helpers used within the realtime model.
  - src/models/tfgridnet_orig/tfgridnet.py: original non‑streaming TF‑GridNet; includes `EmbedTFGridNet`.
  - src/models/tfgridnet_orig/stft.py, src/models/tfgridnet_orig/stft_decoder.py: STFT helpers (legacy/original model path).
  - src/models/config.json: example/reference model hyperparameters.

- Datasets and data generation
  - src/datasets/MixLibriSpeech.py: JAMS‑driven dataset with binaural rendering.
  - src/datasets/MixLibriSpeechDM.py: dynamic mixing at load time.
  - src/datasets/MixLibriSpeechNoisyEnroll.py: noisy enrollment dataset variant.
  - src/datasets/MixLibriSpeechNoisyEnrollNorm.py: noisy enrollment + additional stats (angular velocity, enrollment error, anechoic enrollments).
  - src/datasets/MixLibriSpeechNoisyEnrollDirectional.py / MixLibriSpeechNoisyEnrollWP.py / MixLibriSpeechMotion.py: alternative dataset flavors (direction loss, varied noise/augmentation, motion‑focused).
  - src/datasets/multi_ch_simulator.py: HRTF/BRIR convolution backends (CIPIC/APL/RRBRIR/ASH/CATT/PRA, plus `MultiChSimulator`).
  - src/datasets/motion_simulator.py: moving‑source renderer (ctypes to a compiled simulator) and random path generators.
  - src/datasets/generate_jams.py: generate Scaper JAMS annotations for LibriSpeech‑based mixtures.
  - src/datasets/librispeech2scaper.py: convert LibriSpeech into Scaper‑friendly layout.
  - src/datasets/augmentations.py: white/pink/brown noise generators for augmentation.
  - src/datasets/dvector_embeddings.py: precompute Resemblyzer d‑vectors for utterances.
  - src/datasets/nemo_embed.py: alternative speaker embedding extraction via NVIDIA NeMo (TiTANet).
  - src/datasets/SpeechSeparationDataset.py, src/datasets/OracleMixLibriSpeech.py: utility/legacy datasets for separation and oracle references.

- Losses and evaluation
  - src/losses/LossFn.py: factory wrapper for multiple loss types (SNR/SI‑SDR/SD‑SDR, PIT, mel/perceptual, fused) for quick experimentation.
  - src/losses/sisdr_with_sum_loss.py, src/losses/sisdr_with_pit.py, src/losses/scale_dependent_snr_loss.py, src/losses/perceptual_losses.py, src/losses/fused_loss.py: individual loss implementations.
  - src/eval/binaural.py: ILD/ITD computations with optional moving‑source chunking.

- Evaluation utilities
  - src/ts_hear_test.py: end‑to‑end evaluation producing a CSV with SI‑SNRi, cosine similarity to the ground‑truth embedding, and sample metadata.
  - runs/*/config.json: persisted configs of runs (for reproducibility).


Training Details and Tips

- Loss choice: default target‑hearing setup uses `-SNR` to directly improve intelligibility. Config‑switchable alternatives exist in src/losses if you want stronger scale invariance (SI‑SDR) or perceptual bias (mel/CDPAM/fused).
- Batch sizing: configs divide `batch_size` across GPUs (per‑GPU batch) in src/trainer.py; adjust `eval_batch_size` separately for val/test.
- HRTF diversity: training on multiple HRTF collections via `MultiChSimulator` aids generalization; ensure your `hrtf_list` paths are valid.
- Noisy enrollments: using the noisy‑enrollment datasets builds robustness to real capture conditions; augmentation can be toggled/adjusted.
- Streaming: `stft_pad_size` controls lookahead; increasing it gives context but adds latency. `chunk_causal` enforces chunk‑causal attention.


Inference

- Minimal example (with configs embedded in src/ts_hear_test.py):
  - `python src/ts_hear_test.py --device 0`
  - Produces per‑sample metrics and an aggregated CSV in the run directory.
- External use: load `configs/tsh.json`, create the PLModule, `model.eval().to(device)`, prepare a [B, 2, T] mixture and a [B, 1, E] embedding, and call the module’s `forward` (or the wrapped model with `init_buffers` for chunked streaming; see src/models/tfgridnet_realtime/net.py).


Notes

- Data paths: configs assume a local dataset layout under `data/...`; adapt to your environment.
- W&B: set `WANDB_*` environment variables or log in before training. Runs/IDs are stored under the specified `--run_dir`.
- Reproducibility: Lightning seed is set to 42; data generation uses per‑sample seeded randomness for train/val/test consistency.


License

See LICENSE.
