# Sign Language Recognition — Bridging the Silence 🤟

> A real-time American Sign Language (ASL) alphabet recognizer built with computer vision and deep learning.
> Built to explore how technology can make sign language more accessible.

**Made by Ghanshyam Paunikar**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00897b.svg)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🎯 Live Demo

Here's the system recognizing ASL letters in real time on my webcam:

<p align="center">
  <img src="demo_A.png" width="49%" alt="Letter A — closed fist">
  <img src="demo_D.png" width="49%" alt="Letter D — index finger up">
</p>
<p align="center">
  <img src="demo_F.png" width="49%" alt="Letter F — thumb-index circle, 3 fingers up">
  <img src="demo_I.png" width="49%" alt="Letter I — only pinky up">
</p>
<p align="center">
  <img src="demo_U.png" width="49%" alt="Letter U — index and middle fingers together">
  <img src="demo_W.png" width="49%" alt="Letter W — three fingers up">
</p>
<p align="center">
  <img src="demo_Y.png" width="60%" alt="Letter Y — thumb and pinky extended">
</p>

**What you're seeing:** MediaPipe detects 21 landmarks on my hand in real time (the green dots). My code then analyzes the geometric relationships between those landmarks to classify the ASL letter. The prediction and confidence appear in the top-left corner.

---

## 💡 Why I Built This

The hearing world is loud. The Deaf world is rich with language — but most hearing people can't access it. I wanted to explore whether modern computer vision could help close that gap, even a little.

This isn't a finished translation product. It's a working prototype that shows what's possible — and an honest exploration of the gap between "possible" and "easy."

### The Numbers That Made Me Care

- **430 million people** worldwide live with disabling hearing loss — projected to reach **700 million by 2050** *(World Health Organization)*
- **70+ million deaf people** belong to signing communities globally *(World Federation of the Deaf)*
- **300+ distinct sign languages** exist worldwide — ASL, BSL, ISL, and hundreds more, each a complete language with its own grammar *(United Nations)*
- **Only 40% of national sign languages** are legally recognized by their governments *(WFD)*
- **80% of deaf people** live in developing countries, where interpreter access is limited or non-existent *(WFD)*
- **Unaddressed hearing loss costs the global economy US $980 billion annually** *(WHO)*

### The Research Reality

Building sign language recognition is **hard**. I want to be honest about why:

1. **Data scarcity** — the largest public ASL dataset (WLASL) has only ~12,000 videos. For comparison, ImageNet has 14 million images. Small datasets produce brittle models.
2. **Data quality** — WLASL videos were scraped from YouTube years ago. About 50% are now dead links or taken down. The dataset is noisy, inconsistent, and shrinking.
3. **Signer variation** — ASL signs vary by region, signer speed, body type, and lighting. A model trained on one signer often fails on another.
4. **Non-manual markers** — real ASL uses facial expressions, eyebrow movement, and body tilt for grammar. Most models ignore all of this.
5. **No commercial incentive** — unlike speech recognition, sign language AI lacks the billion-dollar investment that drove Siri and Alexa. Open-source research is the main driver.

This project doesn't solve those problems. But understanding them was half the work.

---

## 🧠 How It Works

```
┌─────────────────┐
│   Webcam Feed   │
│  (live 30 FPS)  │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│   MediaPipe Hands    │       Extracts 21 hand keypoints per frame
│  (21 landmarks × 3D) │       in real time. Pre-trained by Google.
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Normalize Landmarks │       Center on wrist, scale-invariant so
│   (wrist-centered)   │       it works at any hand distance.
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Classification:    │
│                      │
│  (A) Geometric Rules │       Measures finger extension and spread.
│      Built-in, fast  │       Works out of the box, 10+ letters.
│                      │
│  (B) Neural Network  │       Optional: small MLP learns your
│      (optional)      │       specific hand for all 26 letters.
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Temporal Smoothing  │       10-frame majority vote so occasional
│  (majority vote)     │       bad frames don't flicker predictions.
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   On-Screen Display  │       Green skeleton overlay + predicted
│    (OpenCV window)   │       letter + confidence %.
└──────────────────────┘
```

### Why Landmarks Beat Raw Pixels

A raw webcam frame is **224 × 224 × 3 = 150,528 pixels**. A hand landmark vector is just **63 numbers**. Landmarks are:
- **2,000× more compressed** — far less data for the model to learn from
- **Invariant** to lighting, skin color, background, camera angle
- **Already trained** — MediaPipe's hand detector was pre-trained by Google on millions of hands

This is why a simple geometric rule-set or tiny neural net can outperform what a giant CNN would struggle with on raw pixels.

---

## 🚀 Quick Start

### Requirements

- macOS, Linux, or Windows
- Python 3.12 *(Python 3.13 and 3.14 don't support MediaPipe yet — use 3.12)*
- A webcam

### Install

```bash
git clone https://github.com/GhanshyamPaunikar/asl-recognition.git
cd asl-recognition
pip install -r requirements.txt
```

### Run the Alphabet Recognizer (works immediately)

```bash
python asl_alphabet.py
```

A webcam window opens. Show your hand and watch it detect letters. **Press Q to quit.**

### Letters It Recognizes Out Of The Box

| Letter | Hand Shape |
|--------|------------|
| **A** | Closed fist |
| **B** | 4 fingers up, thumb folded across palm |
| **D** | Only index finger up |
| **F** | Thumb + index touching, other 3 up 👌 |
| **I** | Only pinky up |
| **L** | Thumb + index finger in an L shape |
| **U** | Index + middle fingers together, pointing up |
| **V** | Peace sign ✌️ |
| **W** | Index + middle + ring fingers up |
| **Y** | Thumb + pinky out 🤙 |

### Train It On Your Own Hand (for all 26 letters)

```bash
python asl_alphabet.py --train
```

1. The webcam opens
2. Form a letter with your hand
3. Press that letter key on your keyboard — it records 20 samples automatically
4. Repeat for each letter you want
5. Press **Space** to train and save
6. Re-run with your trained model:
   ```bash
   python asl_alphabet.py --model my_dataset.pt
   ```

The trained version is much more accurate because it's calibrated to your specific hand.

---

## 🔮 Future Exploration

Things I want to build into this next:

- **Continuous sign recognition** — string letters into words, words into sentences
- **Real WLASL training** — replace the synthetic-data word model with one trained on real signing videos
- **Non-manual markers** — use MediaPipe Holistic (face + body) to capture facial expressions that ASL grammar depends on
- **ISL (Indian Sign Language) support** — because I'm Indian, and representation in data matters. ISL is severely under-documented compared to ASL.
- **Mobile deployment** — export to ONNX / CoreML / TFLite so this can run on a phone without a laptop
- **Two-hand signs** — most of ASL uses both hands; currently I only track one
- **Speech output** — speak the detected letter/word aloud for accessibility in both directions
- **Streamlit / Gradio web demo** — so people can try it in a browser without installing anything

---

## 🛠️ Technology Used

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.12 | Widest ML ecosystem |
| Hand detection | MediaPipe | Pre-trained, real-time, cross-platform |
| Deep learning | PyTorch | Flexible, research-friendly |
| Video / Image | OpenCV | Industry standard |
| Training | Google Colab (free T4 GPU) | Accessible to anyone |

---

## ⚠️ Honest Limitations

I want this README to age well, so I'm being upfront:

- The **alphabet recognizer** reliably works on ~10 letters with clear geometric signatures. Letters like **C, E, M, N, O** need the training mode because their finger positions are too subtle for simple rules.
- The **word-level Transformer** was trained on synthetic landmark sequences as a pipeline proof-of-concept. Predictions on a real webcam will not be meaningful until retrained on actual WLASL data.
- The system currently only handles **isolated signs**, not continuous signing.
- I trained and tested on my own webcam setup. Your results will depend on lighting, camera quality, and whether MediaPipe can clearly see your hand.

The pipeline is solid. The data is the bottleneck. That's the honest state of this field in 2026.

---

## 🙏 Acknowledgments

- **Google MediaPipe** — for making hand tracking accessible to everyone
- **The WLASL dataset authors** (Li et al., WACV 2020) — for open-sourcing the only serious word-level ASL dataset
- **The Deaf community** — for whom this technology should be built *with*, not just *about*

---

## 👤 Author

**Ghanshyam Paunikar**

I built this to learn, to explore a cause I care about, and to see how far a single developer can push accessibility technology with modern open-source tools.

If you found this useful, consider:
- ⭐ Starring the repo
- Opening an issue with feedback
- Contributing improvements

---

## 📜 License

MIT License — see [LICENSE](LICENSE) file.
Copyright © 2026 Ghanshyam Paunikar.
