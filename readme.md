# TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions

<div align="center">

[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/yaolily/Timechat-OmniCaptioner-40K)
[![Benchmark](https://img.shields.io/badge/🤗%20Hugging%20Face-Benchmark-yellow)](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark)

</div>

## 🌟 Overview
**TimeChat-Captioner** is a multimodal model designed to generate detailed, time-aware, and structurally coherent captions for multi-scene videos. It effectively coordinates visual and audio information to provide comprehensive video descriptions.

- **🏠 Model:** [yaolily/TimeChat-Captioner-GRPO-7B](https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B)
- **📚 Train Dataset:** [yaolily/Timechat-OmniCaptioner-40K](https://huggingface.co/datasets/yaolily/Timechat-OmniCaptioner-40K)
- **🏆 Benchmark:** [yaolily/OmniDenseCap-Benchmark](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark)

---

## 🚀 Inference (Quick Start)

You can use the following Python script to generate captions for your videos. 

**Note:** Ensure you have `qwen_omni_utils.py` in your working directory (or accessible in your python path).

```python
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# 1. Configuration
MODEL_ID = "yaolily/TimeChat-Captioner-GRPO-7B"
VIDEO_PATH = "path/to/your/video.mp4"  # <--- Replace with your video path

MAX_PIXELS = 297920
VIDEO_MAX_PIXELS = 297920

def main():
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

if __name__ == "__main__":
    main()
```