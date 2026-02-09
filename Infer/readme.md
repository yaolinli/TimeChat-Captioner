# Inference on OmniDCBench

This guide walks you through running multi-GPU batch inference with TimeChat-Captioner on the [OmniDCBench](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark) benchmark.

## 1. Download the Benchmark Dataset

Download all files from the HuggingFace dataset repository:

```bash
# Make sure you have git-lfs installed
git lfs install

# Clone the dataset (this may take a while due to large video files)
git clone https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark OmniDCBench
```

After downloading, you should see the following files:

```
OmniDCBench/
├── ours_gt_file.json        # Ground-truth annotations (JSONL format)
├── Movie.tar.gz.aa          # Movie videos (split archive, 3 parts)
├── Movie.tar.gz.ab
├── Movie.tar.gz.ac
├── Youtube.tar.gz.aa        # Youtube videos (single part)
├── videos_3min.tar.gz.aa    # Extended 3-min videos (optional, not needed for standard eval)
├── ...
```

## 2. Extract and Organize Videos

Merge the split archives and extract them into a `Video/` directory:

```bash
cd OmniDCBench

# Create the video directory
mkdir -p Video

# Extract Movie videos
cat Movie.tar.gz.* | tar -xzf - -C Video/

# Extract Youtube videos
mkdir -p Video/Youtube
cat Youtube.tar.gz.* | tar -xf - -C Video/Youtube
```

After extraction, the directory structure should look like:

```
OmniDCBench/
├── ours_gt_file.json
├── Video/
│   ├── Movie/
│   │   ├── 6965768652251628068/
│   │   │   ├── 1_4s_64s.mp4
│   │   │   ├── 1_65s_128s.mp4
│   │   │   └── ...
│   │   └── ...
│   └── Youtube/
│       ├── zFVfTViaMG8/
│       │   ├── 6_1s_66s.mp4
│       │   └── ...
│       └── ...
```


## 3. Configure the Inference Script

Open `Infer/infer.sh` and adjust the following variables to match your setup:

| Variable | Description | Example |
|---|---|---|
| `MODEL_PATH` | Path to the model checkpoint (local path or HuggingFace ID) | `../ckpts/TimeChat-Captioner-GRPO-7B` |
| `INPUT_PATH` | Path to the JSONL ground-truth file | `OmniDCBench/ours_gt_file.json` |
| `VIDEO_DIR` | Root directory containing `Movie/` and `Youtube/` | `OmniDCBench/Video` |
| `OUTPUT_DIR` | Directory to save prediction results | `result/TimeChat-Captioner-GRPO-7B_infer` |
| `GPU_NUM` | Number of GPUs to use for parallel inference | `8` |
| `GPU_START` | Index of the first GPU (e.g., `0` for GPU 0,1,...,N-1) | `0` |

## 5. Run Inference

We recommend using a GPU with at least 60GB of memory for inference on 60-second video clips.


```bash
conda activate timechatcap
cd Infer
bash infer.sh
```

The script will:
1. Split the dataset evenly across `GPU_NUM` GPUs.
2. Launch one inference worker per GPU in parallel.
3. Save per-GPU results to `rank_0.jsonl`, `rank_1.jsonl`, etc.
4. Merge all results into `merged_result.jsonl` after completion.

**Resume support:** If the script is interrupted, simply re-run it. Already-completed items (identified by `clip_path`) will be automatically skipped.

Logs for each GPU worker are saved in `<OUTPUT_DIR>/logs/gpu*.log`.

## 6. Output Format

Each line of the output JSONL file contains the original ground-truth fields plus:

- `prediction`: The raw model output (string).
- `prediction_json` *(optional)*: Parsed structured JSON if the model output is valid JSON.

Example:

```json
{
  "clip_path": "Movie/6965768652251628068/1_4s_64s.mp4",
  "duration": 60,
  "prediction": "[{\"timestamp\": \"00:00-00:15\", \"segment_detail_caption\": \"...\", ...}]",
  "prediction_json": [{"timestamp": "00:00-00:15", "segment_detail_caption": "...", ...}]
}
```
