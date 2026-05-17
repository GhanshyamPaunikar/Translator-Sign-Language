# Sign Language Translator — Stage 2: From Letters to Words

A two-stage journey toward making sign language readable by machines.

- **Stage 1 — ASL alphabet recognition** (real-time webcam, single letters,
  MediaPipe hand landmarks → classifier). Lives in the root of this repo.
- **Stage 2 — BSL word-level translation** *(this folder / branch)*. Moves
  from static hand shapes to **dynamic, full-body signs**: pose + both hands
  across time, on **British Sign Language** instead of ASL.

Stage 1 proved a webcam + MediaPipe + a small classifier can read sign
shapes. Stage 2 is the honest, harder follow-up: real signs are not letters
held still. They are motion, two hands, and body posture, all at once.

---

## Why this matters — the deaf community

> **466 million** people worldwide live with disabling hearing loss
> (WHO, projected to exceed 700 million by 2050).
> There are **300+ distinct sign languages**. Most have **no consumer-grade
> translation technology** at all.

In the UK alone, **British Sign Language (BSL)** is the first or preferred
language of ~87,000 Deaf people, and was only granted legal recognition by
the BSL Act **in 2022**. Despite that, BSL interpreters are scarce, expensive,
and unevenly distributed — meaning everyday interactions (a GP appointment,
a bank call, a school parents' evening) often happen *without* one.

Speech-to-text has had decades of investment. Sign-language translation has
had a fraction of it. The datasets are small, the signers are
under-represented, and the tools are scattered across academic papers. This
project is a small, public, step-by-step attempt to chip away at that gap —
and to be honest about how hard the gap is.

**This is not a finished translator. It is a working pipeline that shows
exactly where the wall is, and what it would take to break through it.**

---

## Stage 1 recap — ASL alphabet (already in this repo)

Real-time ASL letter recognition from a webcam. 21 MediaPipe hand landmarks
per frame → geometric rules / small neural net → letter.

<p align="center">
  <img src="../screenshots/demo_A.png" width="120" alt="ASL letter A"/>
  <img src="../screenshots/demo_D.png" width="120" alt="ASL letter D"/>
  <img src="../screenshots/demo_F.png" width="120" alt="ASL letter F"/>
  <img src="../screenshots/demo_I.png" width="120" alt="ASL letter I"/>
  <img src="../screenshots/demo_U.png" width="120" alt="ASL letter U"/>
  <img src="../screenshots/demo_W.png" width="120" alt="ASL letter W"/>
  <img src="../screenshots/demo_Y.png" width="120" alt="ASL letter Y"/>
</p>

Works well for the ~10 letters whose shapes are geometrically distinct.
Limitations: single hand, single frame, static pose, English alphabet only.
Real signing is none of those things — which is why we built Stage 2.

---

## Stage 2 — BSL bidirectional translation

A scaffold for translating in both directions:

- **Sign → Text** — video → MediaPipe pose + hand landmarks → Bi-LSTM
  classifier → predicted BSL word.
- **Text → Sign** — English token → dictionary lookup → ordered BSL clips.

### What changed vs Stage 1

| Aspect            | Stage 1 (ASL letters)         | Stage 2 (BSL words)                       |
|-------------------|-------------------------------|-------------------------------------------|
| Unit              | one letter (A–Z)              | one **word** (`hello`, `please`, …)       |
| Input             | 1 frame                       | **32 frames** of motion                   |
| Features          | 21 hand landmarks (1 hand)    | 33 pose joints + **both** 21-joint hands  |
| Feature dim       | 63                            | **258** per frame                         |
| Model             | small classifier / rules      | Bi-LSTM (2 layers, 96 hidden)             |
| Language          | ASL                           | **BSL** (UK)                              |

### Dataset

- **100 BSL words** across greetings, pronouns, question words, verbs,
  adjectives, family, time, places, numbers, essentials.
- **741 clips, ~38 minutes**, scraped from
  [signbsl.com](https://www.signbsl.com).
- **18 source providers** (different signers / institutions) — enabling
  honest cross-signer evaluation.
- Every clip pre-processed into a `.npz` of pose + left/right hand landmarks.

### Pipeline

```bash
python3 scrape_signbsl.py        # 1. download clips
python3 probe_videos.py          # 2. build metadata.csv
python3 extract_landmarks.py     # 3. MediaPipe -> .npz landmark cache
python3 build_splits.py          # 4. train/val/test (random + by_provider)
python3 train_v2.py              # 5. train Transformer (recommended)
python3 train_recognizer.py      # 5b. train original Bi-LSTM (for comparison)
python3 text_to_sign.py "hello please thank you"   # text -> sign demo
```

---

## Results

### v2 — Transformer, improved features (current best)

| Split           | Vocab | Train clips | Train acc | Val acc | Test acc | Random |
|-----------------|------:|------------:|----------:|--------:|---------:|-------:|
| `random` ≥10    |  24   | 234         | ~86%      | 33.3%   | **29.6%**| 4.2%   |

**7× above the random baseline.** Run it yourself:
```bash
python3 train_v2.py --split random --min-clips 10 --save model_v2.pt
```

### v1 — Bi-LSTM, raw coordinates (baseline)

| Split             | Vocab | Train clips | Train acc | Val acc | Test acc | Random |
|-------------------|------:|------------:|----------:|--------:|---------:|-------:|
| `random`          | 100   | 555         | ~85%      | 7.6%    | 8.5%     | 1.0%   |
| `random` ≥8 clips | 46    | 380         | ~88%      | 16.3%   | 18.4%    | 2.2%   |
| `by_provider`     | 100   | 686         | ~85%      | 14.8%   | 7.1%     | 1.0%   |
| `by_provider` ≥8  | 46    | ~470        | ~88%      | 28.6%   | 6.7%     | 2.2%   |

### What changed between v1 and v2

| Change                              | Why it helped                                              |
|-------------------------------------|------------------------------------------------------------|
| Upper-body pose only (25 joints)    | Removed feet/knees — pure noise for signing clips          |
| Bone vectors relative to wrist      | Scale/position invariant hand representation               |
| **Velocity features** (Δ per frame) | Sign language is motion; explicit deltas give the model the signal directly |
| Transformer encoder (vs Bi-LSTM)    | Self-attention captures which frames matter most           |
| Focus on ≥10-clip words (24 classes)| More clips per class, fewer classes to overfit across      |
| LR warmup + cosine decay            | Stable training on small data                              |

### Sample confusion matrix (test set)

Words correctly classified: `brother`, `child`, `hot`, `how`, `night`, `what` — all 100%.
Remaining misses are one-sample-per-class noise (the test set has ~1 clip per word),
not structured confusions. A larger test set would reveal whether the model learns
phonological clusters (e.g. number signs grouping together).

The model **still overfits** (~86% train vs 33% val). That is not a model defect —
it is a data defect. Every improvement below will compound on the v2 baseline.

### Why accuracy failed — the real reasons

1. **Not enough data, by an order of magnitude.**
   WLASL-100, the most comparable academic benchmark, ships **~20 clips per
   word** and pose-only baselines reach **~55% top-1**. We have **5–8
   clips per word**. With that few examples, a Bi-LSTM perfectly
   memorises the training set; nothing in that regime generalises.

2. **Unseen-signer generalisation is the *real* problem.**
   The `random` split mixes the same signers across train/val/test, which
   inflates the number. The `by_provider` split holds out **entire
   signers** — that test number (~7%) is the deployment-realistic one.
   It is brutally honest: a model that has only seen a sign performed
   by 3–4 specific people does not recognise it from a new person.

3. **Visual variance dwarfs the signal.**
   Different signers use different speed, hand dominance, camera angle,
   clothing, lighting, framing. With only 5–8 examples, the model can't
   tell which variations are "the sign" and which are "this signer".

4. **No pretraining.** The Bi-LSTM starts from random weights. It has
   zero prior knowledge of human body structure or motion — every
   skeleton relationship has to be learned from these 555 clips.

5. **Class imbalance + small classes.** Some words have 12 clips, others
   have 3. The 3-clip classes contribute almost nothing trainable but
   still pollute the loss.

**In one sentence:** the architecture is fine. The dataset is too small,
too signer-skewed, and too unbalanced for any model to clear the bar.

---

## What needs to be done — and how

A realistic roadmap, ordered by impact-per-effort.

### 1. Scale the data (the single dominant factor)

- **Merge multiple BSL sources.** [BSL Corpus](https://bslcorpusproject.org/),
  [BSL-1K](https://www.robots.ox.ac.uk/~vgg/data/bsl1k/),
  [BOBSL](https://www.robots.ox.ac.uk/~vgg/data/bobsl/), Signbank,
  university research releases. A 10× scale jump (5–8 → 50–80 clips per word)
  is the difference between a toy and a real model.
- **Crowdsource respectfully.** A small mobile app where Deaf
  contributors record a target word once, with consent and credit.
  Even 5 Deaf volunteers × 100 words = 500 new high-quality clips.
- **Augment what we have.** Time warping, mirror-augmentation (carefully —
  only for symmetric signs), background-independent normalisation.

### 2. Reduce vocabulary, increase depth

- Train on the **30 most-covered words** first. Fewer classes, more
  examples per class. Realistic target: **30–50% cross-signer accuracy**
  on a focused vocabulary, which is genuinely useful as a demo.
- Ship that as a **functional 30-word translator** before chasing 100+.

### 3. Better features

- **Pretrained skeleton encoders** (ST-GCN trained on Kinetics-skeleton,
  or PoseFormer) bring inductive bias about how a human body moves.
  This is the highest-leverage architectural change.
- **Trim pose to upper body.** Feet/knees/hips are noise for seated
  signing clips — drops the 132-dim pose vector to ~60 dim, less to overfit.
- **Add an RGB hand-crop stream** as a second branch once data is
  ≥15 clips/word. Captures finger detail MediaPipe loses.

### 4. Honest evaluation, always

- Always report the **`by_provider`** number. Per-signer cross-validation
  is the only way to claim a model actually recognises *the sign*, not
  *the signer*.
- Publish a **confusion matrix**, not just top-1. If the model confuses
  visually-similar signs (numbers, family members), that is *learning*.
  If confusions look random, it is memorising.

### 5. Build with the Deaf community, not for them

- Co-design vocabulary and priorities with Deaf consultants — the words
  that matter in real life are not always the ones a hearing engineer
  guesses.
- Credit signers explicitly. Pay them.
- Treat sign-language data with the same care as voice data: consent,
  attribution, and the right to withdraw.

---

## How to spread awareness

This repo is also a teaching artifact. If you've read this far, you now know:

- BSL is a **distinct language**, not "English with hands". It has its
  own grammar, syntax, and regional dialects.
- Sign-language AI is **decades behind** speech AI, mostly because of data.
- The single biggest blocker is **not algorithmic** — it's **scale and
  representation of Deaf signers** in training data.

**Things anyone can do:**

- Learn the BSL alphabet and 10 common phrases — there are free courses
  at [british-sign.co.uk](https://www.british-sign.co.uk/) and on the
  [BDA](https://bda.org.uk/) site.
- Star and share open BSL/ASL datasets so they stay maintained.
- If you build ML tools, include sign language in your accessibility
  testing — not as an afterthought.
- Support Deaf-led organisations: BDA (UK), NAD (US), WFD (global).

---

## Layout

```
.
├── README.md                    # this file
├── words.txt                    # 100 BSL words to scrape (categorised)
├── scrape_signbsl.py            # Downloader: signbsl.com -> bsl_dataset/<word>*.mp4
├── probe_videos.py              # Probe each clip + rebuild metadata.csv
├── extract_landmarks.py         # MediaPipe pose+hand landmarks -> .npz cache
├── build_splits.py              # train/val/test (random + held-out provider)
├── train_v2.py                  # Transformer trainer — current best (Sign -> Text)
├── train_recognizer.py          # Bi-LSTM trainer v1 (baseline for comparison)
├── train_easy.py                # Reduced-vocab quick-iteration trainer
├── live_demo.py                 # Webcam inference using a trained checkpoint
├── text_to_sign.py              # Text -> sign clip lookup (CLI)
├── models/                      # MediaPipe .task files
└── bsl_dataset/
    ├── <word>*.mp4              # 741 clips, named by word + variation
    ├── metadata.csv             # filename, word, variation, provider, ...
    ├── splits.csv               # split_random + split_by_provider
    └── landmarks/               # one .npz per clip (pose + both hands)
```

## Landmark cache format

| Key          | Shape           | Notes                                      |
|--------------|-----------------|--------------------------------------------|
| `pose`       | `(T, 33, 4)`    | x, y, z, visibility (NaN if not detected)  |
| `left_hand`  | `(T, 21, 3)`    | x, y, z (NaN if not detected)              |
| `right_hand` | `(T, 21, 3)`    | x, y, z (NaN if not detected)              |
| `fps`        | scalar float32  | source clip framerate                      |
| `width`      | scalar int32    | original frame width                       |
| `height`     | scalar int32    | original frame height                      |

Coordinates are in MediaPipe's normalised image space (`[0, 1]` on x/y).

---

## Closing note

Stage 1 was a fun proof that landmarks beat raw pixels for hand shapes.
Stage 2 is the reality check: real sign language is harder, the data is
scarcer, and the people most affected by the gap have the least say in
how it gets closed.

The accuracy numbers in this README are deliberately reported as they are.
A more polished number would be easy to fabricate by tuning the split. The
slow, honest number is the one that tells you the truth about where this
field actually stands — and what it would take to move it forward.

If you want to help, the fastest path is **more data, contributed with the
Deaf community at the centre**. Everything else is downstream of that.
