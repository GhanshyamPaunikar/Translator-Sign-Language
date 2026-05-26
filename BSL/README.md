# BSL — A British Sign Language translator (work in progress)

A small, honest attempt at sign-language translation. Started with the ASL
alphabet (one letter, one frame, one hand). Realised that's not really
"sign language" at all — real signing is **motion, two hands, body posture
and facial cues over time** — and rebuilt the whole thing around BSL words.

This repo is the result so far: a working two-way pipeline, a browser demo
you can sign into, and a frank write-up of where the walls are and what it
would take to break through them.

---

## How this started — from letters to words

The first version was an ASL alphabet recogniser. Webcam in, 21 MediaPipe
hand landmarks per frame, a small classifier on top, letter out. It worked
well for the ~10 letters whose shapes are geometrically distinct
(A, D, F, I, U, W, Y, etc.) and was a fun proof that landmarks beat raw
pixels for hand shapes.

But the more I used it, the more obvious it was that **letters aren't how
Deaf people actually communicate**. Nobody fingerspells a conversation. Real
signs are dynamic — they involve both hands, posture, facial expression,
and time. So I scrapped the alphabet approach and started again, this time
on **British Sign Language word-level recognition** plus a basic
**text → sign** path so it works both ways.

---

## What's actually in here

A working scaffold for bidirectional translation:

- **Sign → Text** — webcam clip (~2.5s) → MediaPipe pose + both hands →
  Bi-LSTM classifier → predicted BSL word.
- **Text → Sign** — English word → dictionary lookup → BSL clip(s) from
  signbsl.com played back in order.
- **Browser demo** — open `http://localhost:8000`, click *Record*, sign a
  word, see top-5 predictions live.
- **Training pipeline** — scrape clips, extract landmarks, build splits,
  train, evaluate honestly (held-out signers, not just held-out clips).

### The dataset

- **100 BSL words** across greetings, pronouns, question words, verbs,
  family, time, places, numbers, essentials.
- **741 clips, ~38 minutes**, scraped from [signbsl.com](https://www.signbsl.com).
- **18 source providers** (different signers / institutions) — so we can
  do honest cross-signer evaluation.
- Each clip pre-processed into a `.npz` of pose + left/right hand landmarks.

### Features used by the model

| Stream     | Shape       | Notes                                   |
|------------|-------------|-----------------------------------------|
| Pose       | `(T, 33, 4)`| x, y, z, visibility — upper body only   |
| Left hand  | `(T, 21, 3)`| x, y, z                                  |
| Right hand | `(T, 21, 3)`| x, y, z                                  |
| Sequence   | 32 frames   | resampled to a fixed length              |

After normalising on shoulder midpoint and width, that's a 186-d vector per
frame fed into a 2-layer Bi-LSTM with mixup, heavy augmentation, label
smoothing, and cosine LR.

---

## How well it works — and why it doesn't (honest)

| Split                | Vocab | Train clips | Val acc | Test acc | Random |
|----------------------|------:|------------:|--------:|---------:|-------:|
| `random` full        | 100   | 555         | **16.3%** | **16.0%** | 1.0%   |
| `random` ≥8 clips    | 46    | 380         | **24.5%** | **20.4%** | 2.2%   |
| `by_provider` full   | 100   | 686         | **25.9%** | **17.9%** | 1.0%   |
| `by_provider` ≥8     | 46    | 470         | **42.9%** | (noisy)   | 2.2%   |

That is **~10–18× above random**. The model is clearly learning *something*
about sign structure — but train accuracy is ~80% while held-out test is
20%, which is the classic signature of **severe overfitting from data
scarcity**, not a broken architecture.

### Why accuracy is what it is

1. **Not enough data, by an order of magnitude.** WLASL-100 (the closest
   academic benchmark) ships ~20 clips per word and pose-only baselines
   hit ~55% top-1. We have 5–8 clips per word. With that few examples a
   Bi-LSTM perfectly memorises the training set; nothing generalises.

2. **Unseen-signer generalisation is the real problem.** The `random`
   split mixes the same signers across train/val/test, which inflates
   numbers. The `by_provider` split holds out **entire signers**. That's
   the deployment-realistic number, and it's brutal — a model that has
   only seen a sign performed by 3–4 people doesn't recognise it from a
   new person.

3. **Visual variance dwarfs the signal.** Different signers vary speed,
   hand dominance, camera angle, framing, clothing, lighting. With only a
   handful of examples, the model can't tell which variations are "the
   sign" and which are "this signer".

4. **No pretraining.** The Bi-LSTM starts from random weights. Every
   skeleton relationship has to be learned from these 555 clips.

5. **A single RGB camera throws away information that matters.** More on
   that below.

**In one line:** the architecture is fine. The dataset is too small, too
signer-skewed, and a 2-D webcam loses depth and orientation cues no model
can recover.

---

## What I learned

- **Sign language is not letters held still.** It is motion, two hands,
  body posture, and facial expression all at once. Anything that ignores
  any of those four will look like a toy in real use.
- **Data is the bottleneck, not models.** I tried bigger models, smaller
  models, different splits, augmentation, mixup, contrastive ideas.
  Nothing comes close to the impact of just having 10× more clips.
- **Honest evaluation is non-negotiable.** It is trivially easy to publish
  an inflated number by mixing signers across splits. The only number
  that means anything is the held-out-signer one.
- **The Deaf community has had decades of being talked *about* by hearing
  engineers and not talked *with*.** Any serious sign-language project has
  to be built with Deaf collaborators from day one, not bolted on.
- **Vision alone is not enough.** A single 2-D webcam loses depth, hand
  orientation, finger occlusion, and the small NMM (non-manual markers)
  on the face that change a sign's meaning. Even with infinite training
  data, RGB-only will plateau.

---

## Why hardware matters (vision alone is not enough)

A 2-D camera is the *cheapest* sensor but also the *most lossy*. Things
that would close a large chunk of the gap:

- **Depth cameras** (Azure Kinect, iPhone TrueDepth, Intel RealSense) —
  resolve hand-in-front-of-body occlusion and recover real 3-D joint
  positions that MediaPipe only estimates.
- **Wrist IMUs / smartwatches** — give absolute hand orientation and
  motion that a camera cannot see when one hand is in front of the other.
- **EMG sleeves** (e.g. the kind Meta CTRL-Labs is developing) — read the
  electrical activity of the forearm muscles, which encodes finger
  configuration *directly* without needing the camera to see the fingers.
- **High-FPS cameras** — many BSL handshape transitions happen in 50ms.
  A standard 30 fps webcam smears them.

A realistic next-generation setup is **one RGB camera + one wristband per
hand**. That combination would give pose, fine hand-shape, and orientation
all at once, and is the level of input professional sign interpretation
systems are converging on.

---

## How to use it

### 1. Install

```bash
git clone https://github.com/GhanshyamPaunikar/Translator-Sign-Language.git
cd Translator-Sign-Language/BSL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You will also need the two MediaPipe `.task` files in `models/`:
[pose_landmarker.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)
and [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task).
Drop them in `models/`.

### 2. Run the browser demo (easiest)

```bash
python3 serve_demo.py --model models/recognizer_improved.pt
```

Open <http://localhost:8000>, allow camera access, click **Record**, and
sign a word. Top-5 predictions show on the right. The model knows 46
words — the list is on the page.

### 3. Train from scratch

```bash
python3 scrape_signbsl.py        # download clips (~40 mins)
python3 probe_videos.py          # build metadata.csv
python3 extract_landmarks.py     # MediaPipe -> .npz landmark cache
python3 build_splits.py          # train/val/test
python3 train_improved.py --split by_provider --min-clips 8 --save models/my.pt
```

### 4. Text → Sign (CLI)

```bash
python3 text_to_sign.py "hello please thank you"
```

---

## How it can be improved

Ordered roughly by impact-per-effort:

1. **More data.** Merge multiple BSL sources
   ([BSL Corpus](https://bslcorpusproject.org/), BSL-1K, BOBSL, Signbank).
   A 10× scale jump (5–8 → 50–80 clips per word) is the difference
   between a toy and a real model. Crowdsource respectfully — with
   consent, credit, and pay — from Deaf signers.
2. **Cut the vocabulary, deepen each class.** Train on the top 30
   most-covered words first. Aim for 30–50% cross-signer accuracy on a
   small, useful vocab before chasing 100+.
3. **Pretrained skeleton encoders.** Replace random init with ST-GCN
   trained on Kinetics-skeleton or a self-supervised masked-joint model.
   Highest-leverage architectural change you can make on this dataset.
4. **Multi-modal hardware.** Add a wristband IMU stream. Add a depth
   camera. Either alone is a step-change; together they remove the
   single-camera ceiling.
5. **Continuous recognition.** Real conversations are not isolated words.
   CTC or RNN-T loss for word-sequence translation is the version this
   needs to grow into.
6. **On-device.** Export to ONNX/CoreML so the browser demo becomes an
   iPhone/Android app. That's where this becomes actually useful in
   someone's life.

---

## Please contribute

This is a small, public attempt at a problem that genuinely affects a lot
of people:

- **466 million** people worldwide live with disabling hearing loss
  ([WHO](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss)),
  projected to exceed 700 million by 2050.
- **300+ distinct sign languages**. Most have **no consumer-grade
  translation technology at all**.
- In the UK alone, **~87,000 people** use BSL as their first or preferred
  language. BSL was only granted legal recognition in 2022. Interpreters
  are scarce, expensive, and unevenly distributed — meaning everyday
  interactions (a GP appointment, a school parents' evening, a bank call)
  often happen without one.

Speech-to-text has had decades of investment. Sign-language translation
has had a tiny fraction of it. The single biggest blocker is **not
algorithms — it is scale and representation of Deaf signers in the
training data**.

**Ways anyone can help:**

- Open issues / PRs with bug fixes, new BSL data sources, better models,
  cleaner pipeline code.
- Contribute clips (with consent, credit, and the right to withdraw).
- Integrate hardware: a wristband IMU stream, a depth camera, a
  high-FPS source.
- If you build ML tools, include sign language in your accessibility
  testing — not as an afterthought.
- Support Deaf-led organisations: [BDA](https://bda.org.uk/) (UK),
  [NAD](https://www.nad.org/) (US), [WFD](https://wfdeaf.org/) (global).
- Learn the BSL alphabet and a few common signs:
  [british-sign.co.uk](https://www.british-sign.co.uk/).

If even a fraction of the effort that went into voice assistants went into
sign-language tools, the gap would close fast. The dataset is the hard
part. Everything else is downstream.

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── words.txt                    # 100 BSL words to scrape (categorised)
├── scrape_signbsl.py            # downloader: signbsl.com -> bsl_dataset/*.mp4
├── probe_videos.py              # probe each clip + rebuild metadata.csv
├── extract_landmarks.py         # MediaPipe -> .npz landmark cache
├── build_splits.py              # train/val/test (random + by_provider)
├── train_recognizer.py          # original Bi-LSTM baseline
├── train_improved.py            # improved trainer (mixup, aug, report)
├── train_easy.py                # reduced-vocab quick-iteration trainer
├── live_demo.py                 # webcam inference (OpenCV native window)
├── serve_demo.py                # browser demo HTTP server
├── static/index.html            # browser demo UI
├── text_to_sign.py              # text -> sign clip lookup (CLI)
├── models/
│   ├── pose_landmarker.task     # MediaPipe pose (download yourself)
│   ├── hand_landmarker.task     # MediaPipe hand (download yourself)
│   └── recognizer_improved.pt   # trained 46-word checkpoint
└── bsl_dataset/
    ├── metadata.csv             # filename, word, variation, provider, ...
    ├── splits.csv               # split_random + split_by_provider
    ├── *.mp4                    # downloaded by scrape_signbsl.py (gitignored)
    └── landmarks/               # built by extract_landmarks.py (gitignored)
```

---

## Closing note

This is not a finished translator. It is a working pipeline that shows
exactly where the walls are and what it would take to break through them.
The accuracy numbers above are deliberately reported as they are —
inflating them would be easy and dishonest. The slow, honest number is
the one that tells the truth about where this field actually stands.

If you want to help, the fastest path is **more data, contributed with
the Deaf community at the centre**. Everything else is downstream of that.
