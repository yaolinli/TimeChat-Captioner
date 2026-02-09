# TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions

<div align="center">

[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/yaolily/Timechat-OmniCaptioner-40K)
[![Benchmark](https://img.shields.io/badge/🤗%20Hugging%20Face-Benchmark-yellow)](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark)

</div>

---

## 🌟 Overview

**TimeChat-Captioner** is a multimodal model designed to generate detailed, time-aware, and structurally coherent captions for multi-scene videos. It effectively coordinates visual and audio information to provide comprehensive video descriptions.

- **🌐 Project Page:** [timechat-captioner.github.io](https://timechat-captioner.github.io/) (coming soon)
- **🏠 Model:** [TimeChat-Captioner (7B)](https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B)
- **📚 Train Dataset:** [TimeChatCap-40K](https://huggingface.co/datasets/yaolily/Timechat-OmniCaptioner-40K)
- **🏆 Benchmark:** [OmniDCBench](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark)

<img width="1773" height="714" alt="image" src="https://github.com/user-attachments/assets/4234857e-5ba6-4b6e-bbb7-eabd0eac2244" />


---

## 🚀 Quick Start

Below, we provide simple examples to show how to use TimeChat-Captioner-GRPO-7B with 🤗 Transformers.

### Installation

```bash
conda create -n timechatcap python=3.12
conda activate timechatcap
pip install torch torchvision
pip install transformers==4.57.1
pip install accelerate
pip install flash-attn --no-build-isolation
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-omni-utils[decord] -U
```

### Usage

> **Note:** To annotate high-quality timestamps and captions, limit video input to around 1 minute. Please segment longer videos into around 60-second clips before processing.

```python
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# 1. Configuration
MODEL_ID = "yaolily/TimeChat-Captioner-GRPO-7B"
VIDEO_PATH = "example_video.mp4"  # <--- Replace with your video path

MAX_PIXELS = 297920
VIDEO_MAX_PIXELS = 297920


print(f"🚀 Processing video: {VIDEO_PATH}")

# 2. Load Model & Processor
print("⏳ Loading model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
model.disable_talker()

# 3. Construct Conversation
# The prompt encourages detailed, time-aware audio-visual description.
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated."
            },
            {
                "type": "video", 
                "video": VIDEO_PATH, 
                "max_pixels": MAX_PIXELS, 
                "max_frames": 160, 
                "fps": 2.0,
                "video_max_pixels": VIDEO_MAX_PIXELS
            }
        ],
    },
]

# 4. Process Inputs
print("⚙️  Processing inputs...")
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

inputs = processor(
    text=text, 
    audio=audios, 
    images=images, 
    videos=videos, 
    return_tensors="pt", 
    padding=True, 
    use_audio_in_video=True
)
inputs = inputs.to(model.device).to(model.dtype)

# 5. Generate Description
print("✨ Generating description...")
with torch.inference_mode():
    text_ids = model.generate(
        **inputs, 
        use_audio_in_video=True, 
        return_audio=False,
        thinker_max_new_tokens=9216,
        talker_max_tokens=9216
    )

response = processor.decode(text_ids[0][inputs.input_ids[0].size(0):], skip_special_tokens=True)

print("\n" + "="*50)
print("🎬 VIDEO DESCRIPTION:")
print("="*50)
print(response)
print("="*50)
```

---

## 📊 Inference on OmniDCBench

We provide a multi-GPU batch inference pipeline to evaluate TimeChat-Captioner on the [OmniDCBench](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark) benchmark.

**Step 1.** Download and extract the benchmark videos (see [`Infer/readme.md`](Infer/readme.md) for full instructions):

```bash
# Clone the dataset
git clone https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark OmniDCBench

# Extract videos into Video/ directory
cd OmniDCBench && mkdir -p Video
cat Movie.tar.gz.*   | tar -xzf - -C Video/
mkdir -p Video/Youtube
cat Youtube.tar.gz.* | tar -xf - -C Video/Youtube
```

**Step 2.** Edit `Infer/infer.sh` to set your paths (`MODEL_PATH`, `VIDEO_DIR`, `INPUT_PATH`, `GPU_NUM`, etc.).

**Step 3.** Run inference:

```bash
cd Infer
bash infer.sh
```

Results will be merged into `<OUTPUT_DIR>/merged_result.jsonl`. See [`Infer/readme.md`](Infer/readme.md) for detailed configuration options and output format.

---

## 🔧 Train

Training can be launched using the scripts provided in `Train/script/*.sh`.
Please refer to [`Train/readme.md`](Train/readme.md) for detailed instructions.

---

## 📝 TODOs

- [ ] Upload eval code to calculate SODA_M and F1.
- [ ] Integrate eval code to lmms-eval.

---

## 📖 Citation


