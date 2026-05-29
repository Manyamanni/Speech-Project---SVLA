# 🤖 SVLA — Speech Vision Language Assistant
### Full Research-Accurate Python Demo Project

**Paper:** *SVLA: A Unified Speech-Vision-Language Assistant with Multimodal Reasoning and Speech Generation*  
**Authors:** Ngoc Dung Huynh, Mohamed Reda Bouadjenek, Imran Razzak, Hakim Hacid, Sunil Aryal  
**Institutions:** Deakin University (AU) · MBZUAI (UAE) · Technology Innovation Institute (UAE)  
**Code:** http://github.com/vlm-svla/svla

---

## ✅ What This Project Covers (Everything from the Paper)

| Paper Section | What | Implemented |
|--------------|------|-------------|
| Section 4.1 | Vision Encoder (CLIP-Large-Patch14-336) | ✅ |
| Section 4.2 | Speech Encoder/ASR (SpeechTokenizer) | ✅ |
| Section 4.3 | Multimodal Fusion (bos/boi/boa tokens) | ✅ |
| Section 4.4 | Speech Decoder/TTS (SoundStorm) | ✅ |
| Section 4.5 | Two paradigms: Cross-modal vs Chain-of-Modality | ✅ |
| Table 1 | ASR + TTS benchmarks (LibriSpeech, VCTK) | ✅ (results viewer) |
| Table 2 | IC + VQA (all 4 I/O configs: T→T, T→S, S→T, S→S) | ✅ |
| Section 5.3 | VQA + Image Captioning results | ✅ |
| Section 5.4 | Accent & Speaking Speed ablation | ✅ |
| Section 3 + Appendix A | Dataset statistics (38.2M pre-train, 2.5M SFT) | ✅ |
| Appendix B | Training strategy (3 stages) | ✅ |

---

## 🏗️ Architecture

```
Speech Input Xs              Image Input Xv
      ↓                            ↓
Speech Encoder s(·)     Vision Encoder g(·) [CLIP-Large-Patch14-336]
[SpeechTokenizer]              ↓
50 tokens/sec           Zv ∈ R^(nv × dv)    [256 tokens/image]
      ↓                            ↓
<<speech-i>> tokens     Wv projection → Hv ∈ R^(nv × dh)
      ↓                            ↓
      ←──────── Multimodal Fusion ──────────→

X = [bos, boi, img(Hv), eoi, Zt, boa, Zs, eoa, eos]

                     ↓
         Backbone LLM fθ(x) [Qwen2.5-1.5B]
                     ↓
        Text tokens  OR  Speech tokens Z's
              ↓               ↓
         Text Output    Speech Decoder s⁻¹(·)
                        [SoundStorm-SpeechTokenizer]
                               ↓
                        Speech Waveform X's
```

---

## 📐 Two Paradigms (Section 4.5)

### Cross-modal Instruction (SVLA-2B)
- Directly maps input → speech output
- Faster but less coherent
- CIDEr for I+T→S: **2.0** (poor)

### Chain-of-Modality Instruction (SVLA-2B-Text-Ins) ✅ Best
- Step 1: Generate TEXT response
- Step 2: Structured prompt → "The textual caption is X. Therefore, the audio caption is:"
- Step 3: Convert to speech
- CIDEr for I+T→S: **64.7** (32× better!)

---

## 🎯 Four I/O Configurations (Table 2)

| Config | Input | Output | Best VQAv2 |
|--------|-------|--------|-----------|
| I+T→T | Image + Text | Text | 69.7 |
| I+T→S | Image + Text | Speech | 37.5 |
| I+S→T | Image + Speech | Text | 52.7 |
| I+S→S | Image + Speech | Speech | 29.4 |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Anthropic API key
export ANTHROPIC_API_KEY=your_key_here   # Mac/Linux
set ANTHROPIC_API_KEY=your_key_here      # Windows

# 3. Run
python app.py

# 4. Open browser
# → http://localhost:7860
```

---

## 🌍 Accent & Speed Ablation (Section 5.4)

Matches paper's Figure 3:
| Accent | VQA Accuracy |
|--------|-------------|
| American | Highest |
| British | High |
| Indian | Lower (despite training) |
| Australian | Lower |

Speeds tested: 0.7× → 1.3× (best at 1.0×, worst at 1.3×)

---

## 📊 Dataset Summary (Section 3 + Appendix A)

- **Pre-training:** 34.3M samples, 50.8K speech hours
  - Stage 1: Text-Speech (LibriHeavy 2M samples)
  - Stage 2: Vision-Text-Speech (LAION-COCO + Visual Genome)
- **SFT:** 2.5M samples, 5,102 hours
  - VQAv2, A-OKVQA, GQA, VizWiz, COCO-Caption-2014
  - LibriSpeech + CommonVoice

---

*Made for Speech Project — Topic: SVLA Multimodal AI*
