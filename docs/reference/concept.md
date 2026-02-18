# IR-aware Prior Work Map (Reading Order) + “Usable Parts” for Few-shot Diffusion Distillation VC (Task B)

Kojiro Tanaka — research notes  
Updated: 2026-02-14 (Asia/Tokyo)

This note curates **IR-aware (Impulse Response aware)** prior work and extracts **transferable building blocks** specifically for:

> Teacher diffusion → Student diffusion (step-distilled) → **few-shot** adaptation  
> while separating / controlling **room + mic + coloration** factors (IR).

---

## How to read this list

- **Start** with diffusion-as-prior for dereverb/enhancement (gives you the _math + measurement consistency_ toolkit).
- Then move to **blind joint dereverb + RIR estimation** (gives you _explicit IR operator_ ideas).
- Then read **RIR generation / estimation** (gives you _IR embedding_ and controllable _IR synthesis_).
- Finally, collect **RIR datasets** for augmentation/conditioning.

Each item below has:

- **What it is**
- **Why it matters for few-shot distillation VC**
- **Usable parts** (drop-in ideas)

---

# A. Diffusion as a clean-speech prior (dereverb/enhancement)

These papers treat a diffusion model as a **strong prior** and enforce **measurement consistency** (i.e., keep generated speech consistent with the observed reverberant/noisy signal). This is the cleanest entry point to “IR-aware thinking”.

## 1) SGMSE: Speech Enhancement and Dereverberation with Diffusion-based Generative Models (Richter et al., 2022)

Source: arXiv:2208.05830

- What: Score-based diffusion for speech enhancement; shows sampler choices and **dereverberation** capability.
- Why for VC-distill: Introduces the “**start from observation** + posterior sampling” mindset; step count reductions (tens of steps) are practical.
- Usable parts:
  - “Observation-initialized” reverse process (not pure noise) for stability.
  - Measurement-consistency framing you can adapt for “**dry voice + IR** ≈ observed voice”.

Reference: https://arxiv.org/abs/2208.05830

## 2) Diffusion model-based MIMO speech denoising & dereverberation (Kimura et al., 2024)

- What: Extends SGMSE to multi-input/multi-output dereverb/denoise.
- Why for VC-distill: Gives multi-condition consistency patterns; useful if you ever add multi-mic or multi-view constraints.
- Usable parts:
  - Multi-condition likelihood / consistency constraints as modular terms in your loss.

Reference: https://s.makino.w.waseda.jp/reprint/Makino/kimura24hscma455-459.pdf

---

# B. Blind dereverb + explicit IR operator estimation (closest to “IR-aware”)

This is where IR-awareness becomes explicit: they parameterize reverberation as an operator and estimate it **along the diffusion trajectory**.

## 3) BUDDy: Single-Channel Blind Unsupervised Dereverberation with Diffusion Models (Moliner et al., 2024)

Source: arXiv:2405.04272

- What: Joint blind dereverb + RIR estimation via posterior sampling. Reverberation is parameterized as **subband exponential-decay filters**, updated iteratively during sampling.
- Why for VC-distill: The best “template” for separating IR from content/timbre: **IR parameters are not absorbed into speaker** during adaptation if you expose them.
- Usable parts:
  - **Parametric IR operator** (subband decay) → a compact “IR embedding” alternative.
  - “Estimate IR parameters while denoising” loop (can be adapted as **auxiliary head** on U-Net).

Reference: https://arxiv.org/abs/2405.04272

## 4) (Follow-on writeup / extended) Unsupervised Blind Joint Dereverberation and Room Impulse Response Estimation using Diffusion Models (Moliner et al., 2024)

- What: Additional/extended BUDDy exposition (arXiv HTML entry).
- Why for VC-distill: More detail on joint estimation; useful for implementation decisions (parameterization, update schedules).
- Usable parts:
  - Practical tricks for stability: parameter update schedules, initialization heuristics.

Reference: https://arxiv.org/html/2408.07472v1

---

# C. RIR estimation / generation (build IR embeddings, controllable spaces)

These are the most direct sources for building a **learned IR embedding** (or generating IRs for augmentation/conditioning).

## 5) Gencho: Room Impulse Response Generation from Reverberant Speech and Text via Diffusion Transformers (2026)

Source: arXiv:2602.09233

- What: Diffusion Transformer that predicts **complex-spectrogram RIRs** from reverberant speech; also shows text-to-RIR adaptation.
- Why for VC-distill: Provides a concrete recipe for “**IR encoder → IR latent**” and shows that IR can be generated under weak guidance.
- Usable parts:
  - Treat RIR as **complex spectrogram** target (phase-aware IR representation).
  - IR encoder design choices (speech → IR latent).
  - Optional “text/descriptor conditioning” idea: “small room, bright, short RT60”.

Reference: https://arxiv.org/abs/2602.09233

## 6) Room Impulse Response Generation Conditioned on Acoustic Parameters (Arellano et al., 2025; Dolby + KTH)

Source: arXiv:2507.12136

- What: Generate RIRs conditioned on acoustic parameters (e.g., RT60, distance, direct-to-reverb ratio), not geometry.
- Why for VC-distill: Perfect for **controlled augmentation** and “style tokens” for space; lets you sweep RT60 systematically in evaluation.
- Usable parts:
  - Parameter-only conditioning schema (RT60 etc.) → cheap IR control vector.
  - Great for building an “IR-robust teacher” by randomized parameter sampling.

Reference: https://arxiv.org/abs/2507.12136

## 7) Room Impulse Response Interpolation using Diffusion Models (2025)

Source: arXiv:2504.20625

- What: Uses diffusion to interpolate RIRs (fill missing measurements).
- Why for VC-distill: Practical tool for expanding sparse measured RIR sets into a larger, smoother manifold.
- Usable parts:
  - “RIR augmentation by interpolation” to reduce overfitting to a small IR set.

Reference: https://arxiv.org/html/2504.20625v1

---

# D. Datasets you’ll actually use (RIR / acoustic parameter corpora)

For IR-aware VC experiments you’ll want:

- **real measured RIRs** (not just simulated)
- **varied mic placements and distances**
- (optional) **spatial RIRs** if you later do multi-channel

## 8) Aachen Impulse Response (AIR) Database

- What: Widely used measured RIR set across many rooms.
- Use: Baseline augmentation pool; good for “random-room” training.
  Reference: https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/  
  Mirror: https://www.openslr.org/20/

## 9) BUT ReverbDB (Szöke et al., 2018)

- What: Real RIR + noise + retransmitted speech; includes practical “cook-book” for acquisition.
- Use: Great for realism; paper argues that **a carefully selected small real-RIR set** can match huge simulated sets—important for few-shot regimes.
  Reference paper: https://arxiv.org/abs/1811.06795  
  Dataset page: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database

## 10) ACE Challenge (Acoustic Characterization of Environments) corpus

- What: Focused on acoustic parameter estimation from speech; includes measured AIRs and parameter labels.
- Use: If you want to train/validate an **IR-parameter predictor** (RT60, DRR) as conditioning.
  Reference: https://www.ee.ic.ac.uk/naylor/ACEweb/  
  Zenodo corpus: https://zenodo.org/records/6257551

### Optional (spatial / multi-channel) dataset

- TAU Spatial Room Impulse Response Database (TAU-SRIR DB): https://zenodo.org/records/6408611

---

# E. How to transfer these ideas into “Few-shot Distillation VC (B)”

Below is a concrete mapping from prior-work concepts → implementation modules.

## E1) Make the teacher “IR-robust” first (cheap win)

Do this before you even think about IR embeddings.

**Training augmentation:**

- Convolve clean training speech with random real RIRs (AIR / BUT ReverbDB).
- Random EQ/mic coloration (simple IIR/FIR shelf/peaks).
- Optional: additive ambient noise at low SNRs.

**Why it helps few-shot:** reduces the chance that the few-shot speaker data’s room/mic gets mistakenly absorbed into the speaker adaptation.

## E2) Add an explicit IR path (two viable designs)

### Design 1 (parametric, light): BUDDy-style “subband decay IR parameters”

- Predict a small vector per utterance: θ_IR = [RT60_bands, DRR_bands, ...]
- Feed θ_IR as conditioning tokens to the diffusion U-Net cross-attn.

Pros: very lightweight; good for XPU.  
Cons: may miss fine coloration.

### Design 2 (learned embedding): Gencho-style IR embedding

- IR encoder: reverberant speech → IR latent
- Condition diffusion on IR latent

Pros: captures richer effects.  
Cons: needs careful disentanglement so it doesn’t leak speaker/content.

## E3) During student distillation, keep IR fixed or explicitly conditioned

Recommended distillation loop:

1. Teacher takes (content, F0, speaker, IR_cond) → generates latent
2. Student is trained to match teacher in **v-space** (or x0), plus perceptual constraints
3. Few-shot adaptation updates only **cross-attn K/V LoRA** and/or speaker embedding — **not** the IR pathway

Key idea: **IR pathway is shared** and not overwritten by few-shot learning.

## E4) Evaluation protocol (must-have)

Sweep IR conditions:

- Dry (no RIR)
- Short RT60
- Long RT60
- Bright vs dark EQ coloration
- Distance near vs far

Report:

- Speaker similarity (ECAPA cosine)
- F0 RMSE
- High-band energy retention
- Consonant-region spectral sharpness (e.g., band energy ratio in 4–8 kHz during consonant segments)

---

# F. A practical “read-and-implement” schedule (2 weeks)

## Days 1–3

- Read SGMSE + implement measurement-consistency concept (just as notes).
- Implement RIR augmentation pipeline.

## Days 4–7

- Read BUDDy; implement **parametric IR vector** baseline (RT60/DRR-like).
- Add IR_cond to diffusion conditioning interface.

## Days 8–10

- Read Gencho; implement “IR encoder → embedding” prototype (optional branch).

## Days 11–14

- Student distillation with IR_cond fixed.
- Few-shot adaptation that **does not touch IR pathway**.
- Evaluation sweeps across IR conditions.

---

# G. Quick checklist (implementation guardrails)

- [ ] Teacher converged and IR-robust before distillation
- [ ] Same noise seeds + matched timesteps for distillation
- [ ] v-pred (or x0) + multi-res STFT loss in student training
- [ ] Few-shot updates limited (cross-attn K/V LoRA, speaker embedding)
- [ ] IR path frozen during few-shot adaptation (unless explicitly studying IR transfer)
- [ ] Evaluation includes IR sweeps, not just dry audio

---

## Appendix: Minimal bibliography (the 10 core items)

1. Richter et al. “Speech Enhancement and Dereverberation with Diffusion-based Generative Models” (2022) — https://arxiv.org/abs/2208.05830
2. Kimura et al. “Diffusion model-based MIMO speech denoising and dereverberation” (2024) — https://s.makino.w.waseda.jp/reprint/Makino/kimura24hscma455-459.pdf
3. Moliner et al. “BUDDy” (2024) — https://arxiv.org/abs/2405.04272
4. Moliner et al. (extended) (2024) — https://arxiv.org/html/2408.07472v1
5. Gencho (2026) — https://arxiv.org/abs/2602.09233
6. Arellano et al. “RIR Generation Conditioned on Acoustic Parameters” (2025) — https://arxiv.org/abs/2507.12136
7. “RIR Interpolation using Diffusion Models” (2025) — https://arxiv.org/html/2504.20625v1
8. AIR Database — https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
9. BUT ReverbDB (paper + dataset) — https://arxiv.org/abs/1811.06795 ; https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
10. ACE Challenge corpus — https://www.ee.ic.ac.uk/naylor/ACEweb/ ; https://zenodo.org/records/6257551
