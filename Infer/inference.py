import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse
import os
import json
from pathlib import Path
import random
from tqdm import tqdm
from typing import Dict, Any

random.seed(1234)

MAX_PIXELS = 297920
VIDEO_MAX_PIXELS = 297920


import time

def process_single_video(video_path, model, processor, output_dir=None):
    # print(f"Processing video: {video_path}") 
    if not os.path.exists(video_path):
        print(f"[Warning] Video file not found: {video_path}")
        return None

    try:
        # 记录开始时间
        start_time = time.time()

        # 构建对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": MAX_PIXELS,
                        "max_frames": 160, # 如果觉得太慢，可以尝试改为 80
                        "fps": 2.0,
                        "video_max_pixels": VIDEO_MAX_PIXELS
                    },
                    {
                        "type": "text",
                        "text": "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated."
                    },
                ],
            },
        ]

        # 处理输入
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

        # 记录推理开始时间
        inference_start_time = time.time()

        # 🔧 强制修复配置
        # 1. 修复 Thinker (最关键)
        if hasattr(model.config, "thinker_config") and hasattr(model.config.thinker_config, "text_config"):
            model.config.thinker_config.text_config.use_cache = True
            print("Fixed: model.config.thinker_config.text_config.use_cache set to True")

        # 2. 确保顶层也开启 (以防万一)
        model.config.use_cache = True

        # 3. 如果有 generation_config，也要确保它是对的
        if model.generation_config is not None:
            model.generation_config.use_cache = True

        # 生成描述
        with torch.no_grad():
            text_ids = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=8192, 
                talker_max_tokens=8192,
                use_cache=True,
                # repetition_penalty=1.05 
            )
            
        # 解码输出
        generated_ids = text_ids[0][inputs.input_ids[0].size(0):]
        response = processor.decode(generated_ids, skip_special_tokens=True)

        # --- 新增：统计与日志 ---
        end_time = time.time()
        total_time = end_time - start_time
        inference_time = end_time - inference_start_time
        
        # 统计长度
        char_count = len(response)
        word_count = len(response.split()) # 简单的空格分词统计
        token_count = len(generated_ids)   # 实际生成的 Token 数量

        # 计算生成速度 (Tokens per second)
        speed = token_count / inference_time if inference_time > 0 else 0

        print(f"✅ [Completed] {os.path.basename(video_path)}")
        print(f"   - Time Taken: {total_time:.2f}s (Inference: {inference_time:.2f}s)")
        print(f"   - Output Length: {word_count} words / {char_count} chars / {token_count} tokens")
        print(f"   - Speed: {speed:.2f} tokens/s")
        print("-" * 50)
        # -----------------------

        return response

    except Exception as e:
        print(f"[Error] Failed processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# def process_single_video(video_path, model, processor, output_dir=None):
#     print(f"Processing video: {video_path}")
#     if not os.path.exists(video_path):
#         print(f"[Warning] Video file not found: {video_path}")
#         return None

#     try:
#         # 构建对话
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": video_path,
#                         "max_pixels": MAX_PIXELS,
#                         "max_frames": 160,
#                         "fps": 2.0,
#                         "video_max_pixels": VIDEO_MAX_PIXELS
#                     },
#                     {
#                         "type": "text",
#                         "text": "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated."
#                     },
#                 ],
#             },
#         ]

#         # 处理输入
#         print("Processing multimodal inputs...")
#         text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
#         print(text)

        
#         audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

#         inputs = processor(
#             text=text,
#             audio=audios,
#             images=images,
#             videos=videos,
#             return_tensors="pt",
#             padding=True,
#             use_audio_in_video=True
#         )
#         inputs = inputs.to(model.device).to(model.dtype)

#         # 生成描述
#         print("Generating description...")
#         # text_ids = model.generate(
#         #         **inputs,
#         #         use_audio_in_video=True,
#         #         return_audio=False,
#         #         thinker_max_new_tokens=8192,
#         #         talker_max_tokens=8192
#         #     )
#         # 关键优化：使用 no_grad 减少显存占用并加速
#         with torch.no_grad():
#             text_ids = model.generate(
#                 **inputs,
#                 use_audio_in_video=True,
#                 return_audio=False,
#                 thinker_max_new_tokens=8192, # 建议：降至 4096 或 2048 防止死循环耗时过长
#                 talker_max_tokens=8192,
#                 repetition_penalty=1.1       # 建议：增加重复惩罚，防止模型陷入复读机死循环
#             )
#         response = processor.decode(text_ids[0][inputs.input_ids[0].size(0):], skip_special_tokens=True)

#         print(f"Generated description for {video_path}")
#         return response

#     except Exception as e:
#         print(f"[Error] Failed processing {video_path}: {e}")
#         return None


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Omni on a video with configurable paths.")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to the input video file (for single video mode).")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to a file listing video JSON lines (with 'clip_path' field).")
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated captions.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.disable_talker()

    # 单视频模式
    if args.video_path:
        response = process_single_video(args.video_path, model, processor, args.output_dir)
        if response:
            print(response)
        return

    # 批量模式
    if not args.input_path or not os.path.isfile(args.input_path):
        raise ValueError(f"--input_path must be a valid file (got {args.input_path})")

    with open(args.input_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    start = args.start_index
    end = args.end_index if args.end_index is not None else len(data)
    data = data[start:end]

    output_path = os.path.join(args.output_dir, f"rank_{args.rank}.jsonl")
    print(f"Processing {len(data)} videos from index {start} to {end} (rank={args.rank})")
    print(f"Output will be saved to: {output_path}")


    # ==========================================
    # 新增逻辑：读取已存在的输出文件以构建缓存
    # ==========================================
    processed_clips = {}
    if os.path.exists(output_path):
        print(f"Found existing output file, scanning for completed items: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    c_path = rec.get("clip_path")
                    # 获取预测结果，如果不存在则默认为空字符串
                    pred = rec.get("prediction", "")
                    if len(pred) > 0 and (c_path not in processed_clips):
                        processed_clips[c_path] = pred
                        update = True
                except json.JSONDecodeError:
                    continue
    # ==========================================  
    for i, item in enumerate(tqdm(data, desc=f"Rank {args.rank}")):
        try:
            clip_path = item.get("clip_path")
            # ==========================================
            # 新增逻辑：检查缓存
            # 如果clip_path在缓存中，且预测内容不为空，则跳过
            # ==========================================
            if clip_path in processed_clips:
                prev_pred = processed_clips[clip_path]["prediction"]
                # 检查是否为空内容 (None 或 空字符串 或 纯空白字符)
                if prev_pred and str(prev_pred).strip() != "" and str(prev_pred) != "FAILED":
                    print(f"[Skip] Already processed with valid content: {clip_path}")
                    continue
                else:
                    print(f"[Resume] Found empty or failed prediction for {clip_path}, re-inferencing...")
            # ==========================================
            if args.video_dir and clip_path and not os.path.isabs(clip_path):
                video_path = os.path.join(args.video_dir, clip_path)
            else:
                video_path = clip_path

            if not video_path or not os.path.exists(video_path):
                print(f"[Skip] Missing video: {video_path}")
                continue

            response = process_single_video(video_path, model, processor, args.output_dir)
            item["prediction"] = response if response else "FAILED"

            # 如果能解析为 JSON，则额外存储
            if response:
                try:
                    prediction_json = json.loads(response)
                    if isinstance(prediction_json, list):
                        item["prediction_json"] = prediction_json
                except json.JSONDecodeError:
                    print(f"[Warning] response is not valid JSON for {clip_path}")

            with open(output_path, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[Error] Exception while processing item {i}: {e}")
            continue


if __name__ == "__main__":
    main()
