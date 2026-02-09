"""
Multi-GPU batch inference script for TimeChat-Captioner.

Usage:
  # Single video mode:
  python inference.py --model_path <MODEL_PATH> --video_path <VIDEO_FILE>

  # Batch mode (called by infer.sh for multi-GPU parallel inference):
  python inference.py --model_path <MODEL_PATH> \
      --input_path <GT_JSONL> --video_dir <VIDEO_DIR> \
      --output_dir <OUTPUT_DIR> --start_index 0 --end_index 100 --rank 0
"""

import os
import json
import time
import argparse
import traceback

import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ============================================================
# Global constants for video processing
# ============================================================
MAX_PIXELS = 297920
VIDEO_MAX_PIXELS = 297920

# The prompt used to instruct the model.
PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated."
)


def process_single_video(video_path: str, model, processor) -> str | None:
    """
    Run inference on a single video and return the generated description.

    Args:
        video_path: Path to the input video file.
        model: The loaded Qwen2.5-Omni model.
        processor: The corresponding processor.

    Returns:
        The generated text response, or None if the video is missing / an error occurs.
    """
    if not os.path.exists(video_path):
        print(f"[Warning] Video file not found: {video_path}")
        return None

    try:
        start_time = time.time()

        # -- Build the conversation --
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": MAX_PIXELS,
                        "max_frames": 160,  # reduce to 80 for faster speed
                        "fps": 2.0,
                        "video_max_pixels": VIDEO_MAX_PIXELS,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            },
        ]

        # -- Prepare inputs --
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=True
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        inference_start = time.time()

        # -- Generate --
        with torch.no_grad():
            text_ids = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=8192,
                talker_max_tokens=8192,
                use_cache=True,
            )

        # -- Decode --
        generated_ids = text_ids[0][inputs.input_ids[0].size(0):]
        response = processor.decode(generated_ids, skip_special_tokens=True)

        # -- Log statistics --
        end_time = time.time()
        total_time = end_time - start_time
        inference_time = end_time - inference_start
        token_count = len(generated_ids)
        word_count = len(response.split())
        speed = token_count / inference_time if inference_time > 0 else 0

        print(f"  [Done] {os.path.basename(video_path)}")
        print(f"    Time: {total_time:.1f}s total, {inference_time:.1f}s inference")
        print(f"    Output: {word_count} words, {token_count} tokens")
        print(f"    Speed: {speed:.1f} tokens/s")

        return response

    except Exception as e:
        print(f"[Error] Failed processing {video_path}: {e}")
        traceback.print_exc()
        return None


def load_existing_results(output_path: str) -> set:
    """
    Scan an existing output JSONL file and return a set of clip_paths
    that already have a valid (non-empty, non-FAILED) prediction.
    This enables resuming interrupted inference runs.
    """
    done = set()
    if not os.path.exists(output_path):
        return done

    print(f"[Resume] Scanning existing results: {output_path}")
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                clip = rec.get("clip_path", "")
                pred = rec.get("prediction", "")
                if clip and pred and str(pred).strip() not in ("", "FAILED"):
                    done.add(clip)
            except json.JSONDecodeError:
                continue

    print(f"[Resume] Found {len(done)} already-completed items, will skip them.")
    return done


def main():
    parser = argparse.ArgumentParser(
        description="TimeChat-Captioner inference (single video or batch mode)."
    )
    # -- Model --
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path (or HuggingFace ID) of the model checkpoint.",
    )
    # -- Single video mode --
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Path to a single video file. If set, batch arguments are ignored.",
    )
    # -- Batch mode --
    parser.add_argument(
        "--input_path", type=str, default=None,
        help="Path to the ground-truth JSONL file (each line has a 'clip_path' field).",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Root directory that contains the video files (Movie/, Youtube/, etc.).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save prediction JSONL files.",
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="Start index of the data slice for this worker.",
    )
    parser.add_argument(
        "--end_index", type=int, default=None,
        help="End index (exclusive) of the data slice for this worker.",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Worker rank (used for naming the output file).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------
    print(f"Loading model from: {args.model_path}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.disable_talker()
    print("Model loaded successfully.")

    # -------------------------------------------------------
    # Single video mode
    # -------------------------------------------------------
    if args.video_path:
        response = process_single_video(args.video_path, model, processor)
        if response:
            print("\n" + "=" * 60)
            print("VIDEO DESCRIPTION:")
            print("=" * 60)
            print(response)
            print("=" * 60)
        return

    # -------------------------------------------------------
    # Batch mode
    # -------------------------------------------------------
    if not args.input_path or not os.path.isfile(args.input_path):
        raise ValueError(f"--input_path must point to a valid file (got: {args.input_path})")

    # Read the ground-truth JSONL
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Slice the data for this worker
    start = args.start_index
    end = args.end_index if args.end_index is not None else len(data)
    data = data[start:end]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"rank_{args.rank}.jsonl")

    print(f"Rank {args.rank}: processing {len(data)} items (index {start}..{end})")
    print(f"Output: {output_path}")

    # Load already-completed results for resume support
    done_clips = load_existing_results(output_path)

    for i, item in enumerate(tqdm(data, desc=f"Rank {args.rank}")):
        try:
            clip_path = item.get("clip_path", "")

            # Skip if already completed
            if clip_path in done_clips:
                continue

            # Resolve the full video path
            if args.video_dir and clip_path and not os.path.isabs(clip_path):
                video_path = os.path.join(args.video_dir, clip_path)
            else:
                video_path = clip_path

            if not video_path or not os.path.exists(video_path):
                print(f"[Skip] Video not found: {video_path}")
                continue

            # Run inference
            response = process_single_video(video_path, model, processor)
            item["prediction"] = response if response else "FAILED"

            # Try to parse as structured JSON
            if response:
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, list):
                        item["prediction_json"] = parsed
                except json.JSONDecodeError:
                    pass

            # Append result to output file
            with open(output_path, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[Error] Item {i}: {e}")
            traceback.print_exc()
            continue

    print(f"Rank {args.rank}: inference complete.")


if __name__ == "__main__":
    main()
