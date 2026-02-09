"""Tools for evaluating dense captions.

Reimplements evaluation metrics that agree with open-sourced methods at
https://github.com/ranjaykrishna/densevid_eval/blob/master/evaluate.py
"""
import json
import collections
import logging
import random
import re
import string
import argparse
import numpy as np
import ipdb
# from metrics.cider import Cider
# from metrics.meteor import Meteor
# from metrics.ptbtokenizer import PTBTokenizer
import os
from copy import deepcopy
def convert_uint8_array_to_string(uint8_array):
        return uint8_array.tobytes().rstrip(b'\x00').decode('utf-8')
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

def convert_strings_to_uint8_arrays(str_tensor, max_str_len=None):
        """Convert string numpy array into uint8 arrays to transfer to TPUs.

    Given the input string array, outputs a uint8 tensor with an additional
    dimension at the end with the size of max_str_len.

    Args:
        str_tensor: The input string array.
        max_str_len: The maximum number of characters to keep in the converted uint8
            array. If None, it is set to the longest string length in the input array.
    Returns:
        Converted uint8 numpy array with an additional dim of size max_str_len.
    """
        # Make sure that the input str_tensor is an np.ndarray of bytes not of object.
        # An object array stores pointers only whereas a bytes array stores actual
        # string bytes
        str_tensor = np.array(str_tensor, dtype=bytes)
        uint8_tensor = np.frombuffer(str_tensor,
                                                                 np.uint8).reshape(str_tensor.shape + (-1, ))
        if max_str_len:
                to_pad = max(0, max_str_len - uint8_tensor.shape[-1])
                uint8_tensor = np.pad(uint8_tensor[..., :max_str_len],
                                                            [[0, 0]] * str_tensor.ndim + [[0, to_pad]])

        return uint8_tensor


def random_string(string_length):
        """Random string generator for unmatched captions."""
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(string_length))


def chased_dp_assignment(scores):
    """Run dp matching as https://github.com/fujiso/SODA/blob/master/soda.py."""

    m, n = scores.shape
    dp = -np.ones((m, n))
    path = np.zeros((m, n))

    def transition(i, j):
            if dp[i, j] >= 0:
                    return dp[i, j]
            elif i == 0 and j == 0:
                    state = [-1, -1, scores[i, j]]
            elif i == 0:
                    state = [-1, transition(i, j - 1), scores[i, j]]
            elif j == 0:
                    state = [transition(i - 1, j), -1, scores[i, j]]
            else:
                    state = [
                            transition(i - 1, j),
                            transition(i, j - 1),
                            transition(i - 1, j - 1) + scores[i, j]
                    ]
            dp[i, j] = np.max(state)
            path[i, j] = np.argmax(state)
            return dp[i, j]

    def get_pairs(i, j):
            p = np.where(path[i][:j + 1] == 2)[0] #TODO: why == 2?
            # pylint: disable=g-explicit-length-test
            if i != 0 and not len(p):
                    return get_pairs(i - 1, j)
            elif i == 0 or p[-1] == 0:
                    return [(i, p[-1])]
            else:
                    return get_pairs(i - 1, p[-1] - 1) + [(i, p[-1])]

    n, m = scores.shape
    max_score = transition(n - 1, m - 1)
    pairs = get_pairs(n - 1, m - 1)
    return max_score, pairs


def iou(interval_1, interval_2):
        """Compute the IOU between two intervals.

    Args:
        interval_1: A tuple (start, end) containing the first interval.
        interval_2: A tuple (start, end) containing the second interval.

    Returns:
        The IOU of the two intervals.
    """
        start_1, end_1 = float(min(*interval_1)), float(max(*interval_1))
        start_2, end_2 = float(min(*interval_2)), float(max(*interval_2))

        intersection = max(0, min(end_1, end_2) - max(start_1, start_2))
        union = min(
                max(end_1, end_2) - min(start_1, start_2),
                end_1 - start_1 + end_2 - start_2)
        result = float(intersection) / (union + 1e-8)
        return result


############################################################
# ✅ Checklist Score (Gemini 2.5 Pro Judge) — with caching, retry & per-dimension scores
############################################################
import json
import os
import re
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold, SafetySetting

class Checklist_Score(object):
    DIM_KEYS = [
        "segment_detail_caption",
        "video_background",
        "acoustics_content",
        "shooting_style",
        "speech_content",
        "camera_state",
    ]

    def __init__(self,
                 max_workers: int = 8,
                 log_file: str = "checklist_scorer.log",
                 cache_dir: str = "cache_checklist_judge_gtkeypoints",
                 model_name: str = "gemini-2.5-pro", # "gemini-2.5-pro",
                 credentials: str = "/home/gaohuan03/yaolinli/code/qwen25omni/OmniVideoCaption/project_config/mmu-gemini-caption-1-5pro-86ec97219196.json",
                 eval_model_name: str = None,
                 debug: bool = False):
        self.debug = debug
        self.max_workers = max_workers
        # self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.credentials = credentials
        # ✅ 评测批次名：用于可读缓存文件名
        self.eval_model_name = eval_model_name or time.strftime("run-%Y%m%d-%H%M")
        # ✅ 缓存目录：如果 eval_model_name 不为空，cache_dir/eval_model_name
        if eval_model_name:
            self.eval_model_name = eval_model_name
            self.cache_dir = Path(cache_dir) / eval_model_name
        else:
            self.eval_model_name = time.strftime("run-%Y%m%d-%H%M")
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._write_lock = Lock()
        # --- logging ---
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

        # --- Gemini client ---
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials
        try:
            user_info = json.load(open(self.credentials))
            self.client = genai.Client(vertexai=True, project=user_info["project_id"], location="global")
        except Exception as e:
            logging.error(f"[Gemini Init] fail: {e}")
            raise

        self.config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.001,
            safety_settings=[
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF)
            ],
            seed=42,
        )

        # 暴露给上层：每个样本的维度分
        self.last_dim_scores: Dict[str, Dict[str, Any]] = {}

    # ----------------------- utils -----------------------
    def _norm_text(self, x: str) -> str:
        x = (x or "").strip()
        x = re.sub(r"\s+", " ", x)
        return x

    def _normalize_gt(self, gt_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        将 GT Ann json 中的内容映射为 checklist keypoints：
        返回结构：
        {
            "segment_detail_caption": [...],
            "video_background": [...],
            "acoustics_content": [...],
            "shooting_style": [...],
            "speech_content": [...],
            "camera_state": [...]
        }
        """
        # ✅ 维度 key 映射（注意 video_background_en → video_background）
        MAP = {
            "segment_detail_caption_en": "segment_detail_caption",
            "video_background_en": "video_background",
            "acoustics_content_en": "acoustics_content",
            "shooting_style_en": "shooting_style",
            "speech_content_en": "speech_content",
            "camera_state_en": "camera_state",
            "segment_detail_caption": "segment_detail_caption",
            "video_background": "video_background",
            "acoustics_content": "acoustics_content",
            "shooting_style": "shooting_style",
            "speech_content": "speech_content",
            "camera_state": "camera_state",
        }

        out = {k: [] for k in self.DIM_KEYS}

        for raw_k, v in gt_item.items():
            if raw_k not in MAP:
                continue

            norm_k = MAP[raw_k]

            # ✅ 过滤：去掉 markdown, **xx**, 换行，生成 keypoints list
            text = self._norm_text(v)
            if not text:
                continue

            # ✅ 按句子拆成 checklist keypoints（你也可以改成更细粒度）
            keypoints = [s.strip() for s in re.split(r"[\.!?]", text) if s.strip()]

            out[norm_k] = keypoints

        return out


    def _hash_key(self, pred_caption: str, gt_dict: Dict[str, List[str]]) -> str:
        """缓存键由 模型名 + caption + 各维度 keypoints（排序后） 组成"""
        parts = [self.model_name, self._norm_text(pred_caption)]
        for k in self.DIM_KEYS:
            lst = [self._norm_text(s) for s in (gt_dict.get(k) or [])]
            lst = sorted([s for s in lst if s])
            parts.append(k + ":" + "|".join(lst))
        raw = "||".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ----------------------- utils -----------------------
    def _safe_id(self, s: str, max_len: int = 120) -> str:
        """将任意字符串转为文件名安全的短 id。"""
        s = (s or "").strip()
        s = re.sub(r"[^\w\-\.]+", "_", s)
        if len(s) > max_len:
            s = s[:max_len]
        return s or "sample"

    def _guess_sample_tag(self, gt_item: Any, pred_caption: str, idx: str) -> str:
        """
        尝试从 GT 抽取 clip id + timestamp 组成样本名。
        优先：clip_path / orig_video_path + timestamp
        否则：pred 文本前 32 字 + 短 hash
        """
        clip = None
        ts = None
        if isinstance(gt_item, dict):
            clip = gt_item.get("clip_path") or gt_item.get("orig_video_path")
            ts = gt_item.get("timestamp") or gt_item.get("ts")
        # 规范化 timestamp
        if isinstance(ts, list) and len(ts) == 2:
            ts_str = f"{ts[0]}-{ts[1]}"
        elif isinstance(ts, str):
            ts_str = ts.replace(":", "")
        else:
            ts_str = None

        parts = []
        if clip:
            parts.append(self._safe_id(Path(str(clip)).stem))
        if ts_str:
            parts.append(self._safe_id(ts_str))
        if not parts:
            head = self._safe_id(pred_caption[:32])
            h = hashlib.sha1(pred_caption.encode("utf-8")).hexdigest()[:8]
            parts = [f"{head}__{h}"]
        parts.append(f"idx{idx}")  # 防止同 clip 同 ts 的并发冲突
        return "__".join(parts)

    def _cache_path_by_tag(self, sample_tag: str) -> Path:
        """缓存文件名：<eval_model_name>__<sample_tag>.json"""
        fname = f"{self._safe_id(self.eval_model_name)}__{self._safe_id(sample_tag)}.json"
        return self.cache_dir / fname

    # -------------------- Gemini call --------------------
    def _build_prompt(self, keypoints_by_dim: Dict[str, List[str]], pred_caption: str) -> str:
        """
        Simplified Gemini judging prompt:
        - GT keypoints are already provided (atomic and accurate)
        - Model only needs to check which ones are mentioned in the predicted caption
        - Output simplified JSON with correct keypoints per dimension
        """
        def fmt_dim(dk):
            return json.dumps(keypoints_by_dim.get(dk, []), ensure_ascii=False)

        prompt = f"""
            You are a **strict evaluator** for fine-grained audio-enhanced video captions.

            You will receive:
            (1) A list of **ground-truth keypoints** already organized in 6 dimensions.
            (2) One **model-generated caption** to evaluate.

            The ground-truth keypoints are already **atomic and accurate**.  
            You only need to check whether each keypoint is **explicitly mentioned or clearly implied** in the model's caption.

            Rules:
            - Mark a keypoint as correct if its meaning appears in the model's caption with the same or equivalent semantics.
            - Ignore differences in phrasing, tense, or minor wording.
            - Do NOT infer or guess beyond the caption content.
            - Do NOT generate new keypoints or summaries.
            - Do NOT output any text other than the required JSON.

            ──────────────────────────────
            📤 Output format (STRICT JSON ONLY):
            {{
            "by_dim": {{
                "segment_detail_caption": {{"correct_keypoints": [<string>, ...], "correct_count": <int>}},
                "video_background":       {{"correct_keypoints": [<string>, ...], "correct_count": <int>}},
                "acoustics_content":      {{"correct_keypoints": [<string>, ...], "correct_count": <int>}},
                "shooting_style":         {{"correct_keypoints": [<string>, ...], "correct_count": <int>}},
                "speech_content":         {{"correct_keypoints": [<string>, ...], "correct_count": <int>}},
                "camera_state":           {{"correct_keypoints": [<string>, ...], "correct_count": <int>}}
            }}
            }}

            ──────────────────────────────
            📥 Ground-truth keypoints (by dimension):
            - segment_detail_caption: {fmt_dim("segment_detail_caption")}
            - video_background:       {fmt_dim("video_background")}
            - acoustics_content:      {fmt_dim("acoustics_content")}
            - shooting_style:         {fmt_dim("shooting_style")}
            - speech_content:         {fmt_dim("speech_content")}
            - camera_state:           {fmt_dim("camera_state")}

            ──────────────────────────────
            🧩 Model-generated caption to evaluate:
            {self._norm_text(pred_caption)}

            Now output only the JSON above, with no explanation.
            """
        if self.debug:
            print("===========Simplified Judge Prompt=============")
            print(prompt)
            print("===============================================")
        return prompt


    def _safe_gemini_json(self, text: str) -> Dict[str, Any]:
        """宽容解析：去掉code fence；只保留最外层 JSON"""
        s = (text or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s).strip()
        # 截取到最后一个大括号
        first, last = s.find("{"), s.rfind("}")
        if first != -1 and last != -1 and last > first:
            s = s[first:last+1]
        try:
            return json.loads(s)
        except Exception:
            return {}

    def _call_gemini_once(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.models.generate_content(
            model=self.model_name, contents=[prompt], config=self.config
        )
        text = "".join(
            p.text for p in resp.candidates[0].content.parts if hasattr(p, "text")
        ).strip()
        return self._safe_gemini_json(text)

    def _call_gemini_retry(self, prompt: str, retries: int = 4, base_delay: float = 1.2) -> Dict[str, Any]:
        for attempt in range(retries):
            try:
                result = self._call_gemini_once(prompt)
                # ✅ 新协议：只强制有 "by_dim"，且每个维度里应有 gt_count / correct_count
                if result and "by_dim" in result:
                    return result
                else:
                    raise ValueError("Judge JSON missing 'by_dim'")
            except Exception as e:
                wait = base_delay * (2 ** attempt)
                logging.warning(f"[Gemini Judge] attempt={attempt+1}/{retries} failed: {e}; sleep {wait:.1f}s")
                time.sleep(wait)
        logging.error("[Gemini Judge] all retries failed.")
        # 兜底：六个维度全 0
        empty = {"gt_count": 0, "gt_keypoints": [], "correct_count": 0, "correct_keypoints": []}
        return {"by_dim": {k: dict(empty) for k in self.DIM_KEYS}}


    # -------------------- public API ---------------------
    def compute_score(self,
                res: Dict[str, List[str]],
                gts: Dict[str, Any],
                sample_names: Dict[str, str] = None
                ) -> Tuple[float, List[int], Dict[str, Dict[str, Any]]]:
        """
        使用简化版 Gemini checklist 输出结构进行评分：
        输出 JSON 仅包含:
        {
        "by_dim": {
            "<dim>": {
            "correct_keypoints": [...],
            "correct_count": int
            }
        }
        }

        返回:
        overall_score: float
        per_sample_correct_counts: List[int]
        last_dim_scores: Dict[str, Dict[str, Any]]
        """
        sample_indices = list(res.keys())
        per_sample_correct = []
        per_sample_total = []
        self.last_dim_scores = {}

        def _job(idx: str):
            pred_caption = self._norm_text(res[idx][0] if res[idx] else "")
            raw_gt = gts[idx]
            # ✅ 直接使用已经存在的 keypoints
            if isinstance(raw_gt, dict) and "keypoints" in raw_gt:
                gt_norm = {dim: raw_gt["keypoints"].get(dim, {}).get("gt_keypoints", [])
                        for dim in self.DIM_KEYS}
            else:
                gt_norm = self._normalize_gt(raw_gt)

            # ✅ 生成 sample_tag
            if sample_names and idx in sample_names:
                sample_tag = self._safe_id(sample_names[idx])
            else:
                sample_tag = self._guess_sample_tag(raw_gt, pred_caption, idx)

            # ✅ 生成缓存路径
            key_hash = self._hash_key(pred_caption=pred_caption, gt_dict=gt_norm)
            cpath = self._cache_path_by_tag(f"{sample_tag}__{key_hash[:12]}")


            # ✅ 调用或读取缓存
            if cpath.exists():
                out = json.load(open(cpath, "r", encoding="utf-8"))
            else:
                prompt = self._build_prompt(gt_norm, pred_caption)
                out = self._call_gemini_retry(prompt)
                out["sample_tag"] = sample_tag
                out["gt_keypoints"] = raw_gt["keypoints"]
                out["pred_caption"] = pred_caption
                with self._write_lock:
                    with open(cpath, "w", encoding="utf-8") as fw:
                        json.dump(out, fw, ensure_ascii=False, indent=4)

            # ✅ 解析 Gemini 输出
            by_dim_raw = out.get("by_dim", {}) or {}
            dim_corrects = {}
            dim_gt_counts = {}

            for dim in self.DIM_KEYS:
                dim_data = by_dim_raw.get(dim, {}) or {}
                correct_count = int(dim_data.get("correct_count", 0))
                gt_count = len(gt_norm.get(dim, []))  # ✅ 直接从 GT 统计
                dim_corrects[dim] = correct_count
                dim_gt_counts[dim] = gt_count

            # ✅ 汇总
            total_gt_kp = sum(dim_gt_counts.values())
            predicted_correct = sum(dim_corrects.values())

            by_dim_ratio = {
                k: (float(dim_corrects[k]) / max(dim_gt_counts[k], 1))
                if dim_gt_counts[k] > 0 else 0.0
                for k in self.DIM_KEYS
            }

            if self.debug:
                ipdb.set_trace()
            return (idx, predicted_correct, total_gt_kp, by_dim_ratio)

        # ✅ 并行执行
        num_workers = min(self.max_workers, len(sample_indices))
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_job, idx) for idx in sample_indices]
            for fut in as_completed(futures):
                try:
                    idx, correct, total_kp, by_dim_ratio = fut.result()
                    per_sample_correct.append(correct)
                    per_sample_total.append(total_kp)
                    dim_scores = {
                        "overall": float(correct) / max(total_kp, 1) if total_kp > 0 else 0.0,
                        "by_dim": by_dim_ratio
                    }
                    self.last_dim_scores[idx] = dim_scores
                except Exception as e:
                    logging.error(f"[compute_score aggregation] {e}")
                    per_sample_correct.append(0)
                    per_sample_total.append(1)

        overall_score = float(sum(per_sample_correct)) / max(sum(per_sample_total), 1)
        return overall_score, per_sample_correct, self.last_dim_scores





def evaluate_detections(predicted_segments, gt_segments, splits=None, iou_thresholds=(0.3, 0.5, 0.7, 0.9)):
    """Compute the mean P/R between the predicted and ground truth segments.

    Args:
        predicted_segments: A numpy array of shape [K x 2] containing the predicted
            segments.
        gt_segments: A numpy array of shape [S x 2] containing the ground truth
            segments.
        splits: A numpy array of shape [S] indicating the annotation set.
        iou_thresholds: The IOU thresholds to use for Precision/Recall calculations.

    Returns:
        precision: The mean precision of the predictions over the IOU thresholds.
        recall: The mean recall of the predictions over the IOU thresholds.
        best_miou: The mIoU.
        iou_matrices: dictionary mapping each split to the corresponding iou matrix.
    """
    # Recall is the percentage of ground truth that is covered by the predictions.
    # Precision is the percentage of predictions that are valid.

    best_recall = []
    best_precision = []
    iou_matrices = {}

    predicted_shape = predicted_segments.shape[0]
    for split in set(splits):
            metrics = {}
            for threshold in iou_thresholds:
                    metrics[str(threshold)] = {
                            'gt_covered': set(),
                            'pred_covered': set(),
                    }
            split_idx = np.where(splits == split)[0]
            split_gt_segments = np.array([gt_segments[idx] for idx in split_idx])

            gt_shape = split_gt_segments.shape[0]

            # Compute the IOUs for the segments.
            iou_matrix = np.zeros((gt_shape, max(predicted_shape, 1)))
            for idx_g, gt_segment in enumerate(split_gt_segments):
                    cur_max_iou = 0
                    for idx_p, segment in enumerate(predicted_segments):
                            sample_iou = iou(segment, gt_segment)
                            iou_matrix[idx_g, idx_p] = sample_iou
                            cur_max_iou = max(cur_max_iou, sample_iou)
                            for threshold in iou_thresholds:
                                    if sample_iou > threshold:
                                            metrics[str(threshold)]['pred_covered'].add(idx_p)
                                            metrics[str(threshold)]['gt_covered'].add(idx_g)

            # Compute the precisions and recalls for each IOU threshold.
            for threshold, m in metrics.items():
                    pred_covered = m['pred_covered']
                    gt_covered = m['gt_covered']

                    # Avoid dividing by 0 for precision
                    m['precision'] = float(len(pred_covered)) / max(
                            float(predicted_shape), 1.0)
                    m['recall'] = float(len(gt_covered)) / float(gt_shape)

            precision = [m['precision'] for m in metrics.values()]
            recall = [m['recall'] for m in metrics.values()]
            if best_precision:
                    best_precision = [
                            max(precision[i], best_precision[i])
                            for i in range(len(precision))
                    ]
                    best_recall = [
                            max(recall[i], best_recall[i]) for i in range(len(recall))
                    ]
            else:
                    best_precision, best_recall = precision, recall
            iou_matrices[int(split)] = iou_matrix

    return best_precision, best_recall, iou_matrices


def match_captions(predicted_segments,
    gt_segments,
    predicted_captions,
    gt_captions,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9)):
    """Matches the predicted captions to ground truth using the IOU thresholds.

    Args:
     predicted_segments: A numpy array of shape [K x 2] containing the predicted
         segment intervals.
     gt_segments: A numpy array of shape [S x 2] containing the ground truth
         segment intervals.
     predicted_captions: A list of string of shape [K] containing the
         corresponding K predicted captions.
     gt_captions: A list of strings of shape [S] containing the corresponding S
         ground truth captions.
     iou_thresholds: A list of thresholds for IOU to average over.

    Returns:
     ground_truths_filtered: Filtered list of ground truth captions for all
        threshold.
     predictions_filtered: Matching list of predicted captions for all
        threshold.
     isxes: For each threshold, contains lists of isx of matches.
    """
    # Setup a set of dictionaries to hold the results.
    ground_truths_filtered = {
            str(threshold): {}
            for threshold in iou_thresholds
    }
    predictions_filtered = {str(threshold): {} for threshold in iou_thresholds}

    # Create GT lists for each of the IOU thresholds.
    isx = 0
    isxes = {str(threshold): [] for threshold in iou_thresholds}
    for idx_p, segment in enumerate(predicted_segments):
            pc_idxp = predicted_captions[idx_p]
            added = {str(threshold): False for threshold in iou_thresholds}
            for idx_g, gt_segment in enumerate(gt_segments):
                    gt_idxg = gt_captions[idx_g]
                    sample_iou = iou(segment, gt_segment)
                    for threshold in iou_thresholds:
                            if sample_iou >= threshold:
                                    key = str(isx)
                                    isxes[str(threshold)].append(isx)
                                    isx += 1
                                    ground_truths_filtered[str(threshold)][key] = [{
                                            'caption':
                                            gt_idxg
                                    }]
                                    predictions_filtered[str(threshold)][key] = [{
                                            'caption':
                                            pc_idxp
                                    }]
                                    added[str(threshold)] = True
            for threshold in iou_thresholds:
                    if not added[str(threshold)]:
                            key = str(isx)
                            isxes[str(threshold)].append(isx)
                            isx += 1
                            # Set this to a random string with no match to the predictions to
                            # get a zero score
                            ground_truths_filtered[str(threshold)][key] = [{
                                    'caption':
                                    random_string(random.randint(10, 20))
                            }]
                            predictions_filtered[str(threshold)][key] = [{
                                    'caption': pc_idxp
                            }]

    return ground_truths_filtered, predictions_filtered, isxes


def horizontal_expand_pairs(iou_cur, pairs):
    """
    Step2: horizontally expand DP pairs so that:
        - multiple predicted segments can belong to one GT segment
        - every pred seg (IoU > 0) is assigned to exactly ONE GT

    Input
        iou_cur: 2D numpy array, shape [S, K]
        pairs: List[(gt_idx, pred_idx)] from chased_dp_assignment

    Output
        merged_groups: List[(gt_idx, [pred_idx_1, pred_idx_2, ...])]
    """
    S, K = iou_cur.shape

    # 统计 DP 覆盖到的 pred 索引
    covered_pred = sorted({pj for (_, pj) in pairs})
    need_expand = (len(covered_pred) < K) # 只有当dp返回的path未覆盖所有pred时，才需要扩展，把所有pred segment都分配给一个gt

    # 把每一行的最右锚点整理成 j_i（没有锚点的行记为 None）
    # 注意：原始 DP 每行至多一个锚点（get_pairs只取 p[-1]）
    row_anchor = [None] * S
    for gi, pj in pairs:
        row_anchor[gi] = pj

    merged_groups = []     # [(gt_idx, [pred_idx, ...]), ...]
    used_pred = set()      # 已并入的 pred，避免重复归属

    # 帮助函数：把 (gi, preds_set) 写入 merged_groups
    def _push_group(gi, preds_set):
        if preds_set:
            merged_groups.append((gi, sorted(preds_set)))

    # 扫描各行：只在 (prev_j, cur_j] 内向左扩展
    prev_j = -1
    for gi in range(S):
        cur_j = row_anchor[gi]
        if cur_j is None:
            # 该行没有锚点，不扩展，只把 prev_j 更新为当前 prev_j（不变）
            continue

        group = set()
        # 先把锚点本身纳入
        group.add(cur_j)
        used_pred.add(cur_j)

        if need_expand:
            # 只向左扩展到上一行锚点（不含）与本行锚点（含）之间
            left_bound = prev_j
            right_bound = cur_j
            for jj in range(left_bound + 1, right_bound + 1):
                if jj in used_pred:
                    continue
                group.add(jj)
                used_pred.add(jj)

            # 右扩展（补齐漏掉的 pred）
            # 寻找下一行的锚点，作为右扩展边界
            next_anchor = None
            for nxt in range(gi + 1, S):
                if row_anchor[nxt] is not None:
                    next_anchor = row_anchor[nxt]
                    break

            # 如果没有下一个锚点，则扩展到最右端 K-1
            right_limit = next_anchor if next_anchor is not None else K
            for jj in range(cur_j + 1, right_limit):
                if jj in used_pred:
                    continue
                group.add(jj)
                used_pred.add(jj)
        _push_group(gi, group)
        prev_j = cur_j

    # 尾巴处理：最后一个锚点右侧的 preds（jj > prev_j）
    # 若希望“每个 pred 都分到某个 gt”，可把这些里 IoU>0 的分配给最后一个有锚点的行
    last_row_with_anchor = None
    for gi in range(S-1, -1, -1):
        if row_anchor[gi] is not None:
            last_row_with_anchor = gi
            break

    if need_expand and last_row_with_anchor is not None and prev_j is not None:
        tail_group_extra = set()
        for jj in range(prev_j + 1, K):
            if jj in used_pred:
                continue
            if iou_cur[last_row_with_anchor, jj] > 0:
                tail_group_extra.add(jj)
                used_pred.add(jj)
        # 把尾巴塞回最后一个已有分组的行（如该行之前为空，则创建）
        if tail_group_extra:
            # 找到该行在 merged_groups 里的位置
            for idx, (gi, preds) in enumerate(merged_groups):
                if gi == last_row_with_anchor:
                    merged_groups[idx] = (gi, sorted(set(preds) | tail_group_extra))
                    break
            else:
                merged_groups.append((last_row_with_anchor, sorted(tail_group_extra)))
    return merged_groups


def soda_m(iou_matrices,
           scorer,
           predicted_captions,
           gt_captions,
           splits,
           iou_thresholds=0.0,
           video_id=None,
           debug=False, 
           predicted_segments=None, 
           gt_segments=None):
    """
    SODA_m (ours): 
        1) 用 IoU 矩阵 DP 做 matching（不乘 caption score）
        2) 针对 pred→gt 的一对多情况进行合并
        3) 一对一调用 caption scorer (Checklist_Score)
        4) 基于 caption 匹配得分计算 P / R / F1
        
    Output:
    {
        split_id: {
            "SODA_m_total": float,
            "SODA_m_by_dim": {
                "segment_detail_caption": float,
                "video_background": float,
                "acoustics_content": float,
                "shooting_style": float,
                "speech_content": float,
                "camera_state": float
            }
        }
    }
    """
    if predicted_captions is None or len(predicted_captions) == 0:
        return {int(split): 0 for split in splits}

    unique_splits = set(splits)
    fs = {}
    for split in unique_splits:
        # 获取该 annotator 对应的 gt + iou matrix
        split_idx = np.where(splits == split)[0]
        split_gt_caps = [gt_captions[idx] for idx in split_idx]
        iou_matrix = iou_matrices[int(split)]     # shape=[S x K]

        best_f1 = 0.0

        threshold = iou_thresholds
        # step 1 — 仅根据 IoU DP 匹配 pred 到 GT（得到锚点）
        iou_cur = np.copy(iou_matrix)
        iou_cur[iou_cur < threshold] = 0.0
        _, pairs = chased_dp_assignment(iou_cur)        # [(gt_idx, pred_idx), ...]
        pairs = sorted(pairs, key=lambda x: (x[0], x[1]))  # 稳妥起见按 (gt, pred) 排序

        # step 2 — 基于锚点做“水平扩展”，形成多对一的 merged_groups
        # 原因：DP 仅返回对角(“配对”)转移，不会把同一 GT 行里“还应该归到它名下”的其它预测时间段也带出来。
        # 示例：
        # iou_cur:
        # [[0.466, 0.533, 0,     0],
        # [0,     0,     0,     0],
        # [0,     0,     1.0,   0],
        # [0,     0,     0, 0.3103]]
        # paths returned:
        # [(0,1), (2,2), (3,3)]
        # extend paths:
        # [(0,0), (0,1), (2,2), (3,3)]

        merged_groups = horizontal_expand_pairs(iou_cur, pairs)
        # pdb.set_trace()
        # 若还有未覆盖且 IoU 全为 0 的 preds，你可以选择保持未匹配，或做一个“最近锚点行”的兜底分配。
        # 这里选择保持未匹配，不再强塞，避免引入噪声。
        # ====== 合并结束，得到 merged_groups 形如：[(0, [0,1]), (2, [2]), (3, [3])] ======
        
        # step 3 — 调 Checklist Score 进行 caption 匹配（只一对一计算）
        caption_scores = { "total": [] }
        for dim in scorer.DIM_KEYS:
            caption_scores[dim] = []
        for gt_idx, pred_group in merged_groups:
            pred_caps = "\n".join([predicted_captions[i] for i in pred_group])
            gt_cap = split_gt_caps[gt_idx]
            
            
            if debug:
                gt_time = gt_segments[gt_idx]
                pred_time = [predicted_segments[pred_idx] for pred_idx in pred_group]
                print("=======================================")
                print(f"\n\n\n===>gt_idx:\t{gt_idx}; \tgt_timestamp:{gt_time}")
                print(f"===>pred_group:\t{pred_group}; \tpred_timestamp:{pred_time}")
                print(f"===>gt_captions:\t\n{gt_cap}\n\n")
                print(f"===>pred_captions:\t\n{pred_caps}\n\n")

            overall_score, _, all_dim_scores  = scorer.compute_score(
                res={ "0": [pred_caps] },
                gts={ "0": gt_cap },
                sample_names={ "0": video_id} if video_id is not None else None,
            )
            
            
            try:
                dim_scores = all_dim_scores["0"]["by_dim"]
            except:
                print("all_dim_scores", all_dim_scores)
            # if debug:
            

            

            # ipdb.set_trace()
            # ✅ 累加每个维度得分
            try:
                for dim, s in dim_scores.items():
                    caption_scores[dim].append(s)
                # ✅ 累加整体 total score
                caption_scores["total"].append(overall_score)
            except:
                print("===> Checklist Score Compute Error!! \tvideo_id", video_id)
                print("===> pred_caps: \n", pred_caps)
                DIM_KEYS = [
                    "segment_detail_caption",
                    "video_background",
                    "acoustics_content",
                    "shooting_style",
                    "speech_content",
                    "camera_state",
                ]
                for dim in DIM_KEYS:
                    caption_scores[dim].append(0.0)
                caption_scores["total"].append(0.0)
                
        # step 4 — 计算 P / R / F1
        n_pred = len(merged_groups)
        n_gt = len(split_gt_caps)
        best_f1 = {}
        for dim, cap_scores in caption_scores.items():
            max_score = sum(cap_scores)
            p = max_score / n_pred if n_pred > 0 else 0
            r = max_score / n_gt
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            best_f1[dim] = f1

        fs[int(split)] = {
            "SODA_m_total": best_f1["total"],
            "SODA_m_by_dim": { dim: best_f1[dim] for dim in scorer.DIM_KEYS }
        }
    return fs

def process_soda(per_video_results):     
    # ----------- ✅ 重点：计算 SODA_m 总分 和 分维度平均分 -----------
    soda_global = {
        "SODA_m_total": 0,
        "SODA_m_by_dim": {
            "segment_detail_caption": 0,
            "video_background": 0,
            "acoustics_content": 0,
            "shooting_style": 0,
            "speech_content": 0,
            "camera_state": 0,
        }
    }

    valid_video_count = 0  # 用来做平均

    for v in per_video_results:
        soda = v.get("SODA_m", None)
        if soda is None:
            continue

        # ✅ 支持两种情况：{0: {...}} 或错误情况 {0: 0}
        soda_obj = soda.get(0, None)
        if isinstance(soda_obj, dict):
            soda_total = soda_obj.get("SODA_m_total", 0)
            soda_dims = soda_obj.get("SODA_m_by_dim", {})
        else:
            # ✅ 如果是 {0: 0}，算合法输入，但是所有值为 0
            soda_total = 0
            soda_dims = {
                "segment_detail_caption": 0,
                "video_background": 0,
                "acoustics_content": 0,
                "shooting_style": 0,
                "speech_content": 0,
                "camera_state": 0,
            }

        valid_video_count += 1
        soda_global["SODA_m_total"] += soda_total

        # 分维度累加
        for dim, val in soda_dims.items():
            soda_global["SODA_m_by_dim"][dim] += val


    # ✅ 求平均
    if valid_video_count > 0:
        soda_global["SODA_m_total"] /= valid_video_count
        for dim in soda_global["SODA_m_by_dim"]:
            soda_global["SODA_m_by_dim"][dim] /= valid_video_count

    return soda_global

def evaluate_dense_captions(valid_gt_vnum,
                            valid_pred_vnum, 
                            predicted_segments,
                            gt_segments,
                            predicted_captions,
                            gt_captions,
                            keys,
                            iou_thresholds=(0.3, 0.5, 0.7, 0.9),
                            soda=True,
                            tmponly=False,
                            max_workers=8,
                            eval_model_name=None,
                            splits=None, debug=False):
    """
    Dense Video Captioning 主入口：
    现在只计算两个指标：
        (1) Detection-level: F1 score（基于 IOU）
        (2) Caption-level: SODA_m（改进版 SODA_c）

    Args:
        predicted_segments: list(np.ndarray [K,2])
        gt_segments:        list(np.ndarray [S,2])
        predicted_captions: list(list[str])
        gt_captions:        list(list[str])
        keys:               list[str], video key
        iou_thresholds:     IOU 阈值
        soda:               是否计算 SODA（我们现在做的是 SODA_m）
        splits:             list(np.ndarray [S])  train / val / test 之类
    Returns:
        metric_tiou: dict{ metric_name -> list[video_score] }
    """

    assert len(predicted_segments) == len(gt_segments) == len(
            predicted_captions) == len(gt_captions) == len(keys), "Length of gt and pred inputs do not match."

    # Handle if these are lists, or single samples.
    assert all(
            [isinstance(p, list) for p in [predicted_segments, gt_segments]])
    # Only construct the scorers once, so that we don't have any issues with
    # overhead when running multiple evaluations.
    scorers = {
            'Checklist_Score': Checklist_Score(max_workers=max_workers, eval_model_name=eval_model_name, debug=debug),
    }

    per_video_results = []   # ✅ store per video result
    num_videos = len(keys)
    # Compute dense video captioning metrics at the video level
    for idx, key in enumerate(keys):
        # 调用 evaluate_single_dense_captions() 对单个视频算：
        # F1 + SODA_m
        split = splits[idx] if splits is not None else None
        result = evaluate_single_dense_captions(
            predicted_segments[idx],
            gt_segments[idx],
            predicted_captions[idx],
            gt_captions[idx],
            key,             # video key
            scorers=scorers,   # ✅ 调用 Checklist_Score() 替换 METEOR
            iou_thresholds=iou_thresholds,
            soda=soda,
            tmponly=tmponly,
            split=split,
            debug=debug,
        )
        # ✅ 保存每个视频的结果
        per_video_results.append(result)
        if tmponly:
            print(f"==> [{idx}/{num_videos}] video {key} results:\n F1 score: {result['F1_Score']}\n\n")
        else:
            print(f"==> [{idx}/{num_videos}] video {key} results:\n F1 score: {result['F1_Score']}\n Soda_m: {result['SODA_m']}\n")
    # ---------------------------------------
    # ✅ Compute corpus-level average results
    # ---------------------------------------
    final_result = {}

    # keys in result dict to average
    avg_keys = [
        'Precision_Mean', 'Recall_Mean', 'F1_Score', 'SODA_m'
    ]

    for k in avg_keys:
        if k == 'SODA_m': 
            continue
        else:
            values = [v.get(k, 0.0) for v in per_video_results]
        final_result[k] = float(sum(values)) / max(len(values), 1)

    final_result["SODA_m"] = process_soda(per_video_results)
    final_result["num_videos"] = num_videos
    final_result["model_name"] = eval_model_name

    # ---------------------------------------
    # ✅ Save logs to results/{model_name}/log
    # ---------------------------------------
    if eval_model_name is not None:
        save_root = f"results_keypoints/{eval_model_name}/log"
        os.makedirs(save_root, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M")
        avg_path = os.path.join(save_root, f"{timestamp}_total_result.json")
        per_case_path = os.path.join(save_root, f"{timestamp}_per_case_result.json")

        # 保存平均结果
        final_result["valid_gt_video_num"] = valid_gt_vnum
        final_result["valid_pred_video_num"] = valid_pred_vnum
        with open(avg_path, "w", encoding="utf-8") as fw:
            json.dump(final_result, fw, indent=4, ensure_ascii=False)

        # 保存每个视频 case 结果
        with open(per_case_path, "w", encoding="utf-8") as fw:
            json.dump(per_video_results, fw, indent=4, ensure_ascii=False)

        
        print(f"✅ Saved corpus avg result to: {avg_path}")
        print(f"✅ Saved per case result to:   {per_case_path}")

    return final_result




def evaluate_dense_captions_para(valid_gt_vnum,
                            valid_pred_vnum, 
                            predicted_segments,
                            gt_segments,
                            predicted_captions,
                            gt_captions,
                            keys,
                            iou_thresholds=(0.3, 0.5, 0.7, 0.9),
                            soda=True,
                            tmponly=False,
                            max_workers=8,
                            eval_model_name=None,
                            splits=None, debug=False):
    """
    Dense Video Captioning 主入口：
    现在只计算两个指标：
        (1) Detection-level: F1 score（基于 IOU）
        (2) Caption-level: SODA_m（改进版 SODA_c）

    Args:
        predicted_segments: list(np.ndarray [K,2])
        gt_segments:        list(np.ndarray [S,2])
        predicted_captions: list(list[str])
        gt_captions:        list(list[str])
        keys:               list[str], video key
        iou_thresholds:     IOU 阈值
        soda:               是否计算 SODA（我们现在做的是 SODA_m）
        splits:             list(np.ndarray [S])  train / val / test 之类
    Returns:
        metric_tiou: dict{ metric_name -> list[video_score] }
    """

    assert len(predicted_segments) == len(gt_segments) == len(
            predicted_captions) == len(gt_captions) == len(keys), "Length of gt and pred inputs do not match."

    # Handle if these are lists, or single samples.
    assert all(
            [isinstance(p, list) for p in [predicted_segments, gt_segments]])
    # Only construct the scorers once, so that we don't have any issues with
    # overhead when running multiple evaluations.
    scorers = {
            'Checklist_Score': Checklist_Score(max_workers=max_workers, eval_model_name=eval_model_name, debug=debug),
    }


    # Compute dense video captioning metrics at the video level
    # ==== 并行版本，保持与 keys 顺序一致 ====
    num_videos = len(keys)
    per_video_results = [None] * num_videos  #  ✅ store per video result

    def _job(i: int):
        key = keys[i]
        split = splits[i] if splits is not None else None
        res = evaluate_single_dense_captions(
            predicted_segments[i],
            gt_segments[i],
            predicted_captions[i],
            gt_captions[i],
            key,                       # video key
            scorers=scorers,           # ✅ 原参数完全保留
            iou_thresholds=iou_thresholds,
            soda=soda,
            tmponly=tmponly,
            split=split,
            debug=debug,
        )
        result = res
        if tmponly:
            print(f"==> [{i}/{num_videos}] video {key} results:\n"
                    f"   F1 score: {result['F1_Score']}\n")
        else:
            print(f"==> [{i}/{num_videos}] video {key} results:\n"
                f"   F1 score: {result['F1_Score']}\n"
                f"   Soda_m:   {result['SODA_m']}\n")
        return i, key, res

    print(f"🔥 Parallel evaluation with max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_job, i) for i in range(num_videos)]
        for fut in tqdm(as_completed(futures), total=num_videos, desc="Checklist Eval"):
            i, key, result = fut.result()
            per_video_results[i] = result  # ✅ 放回原位，顺序不乱
            
    # ---------------------------------------
    # ✅ Compute corpus-level average results
    # ---------------------------------------
    
    final_result = {}

    # keys in result dict to average
    avg_keys = [
        'Precision_Mean', 'Recall_Mean', 'F1_Score', 'SODA_m'
    ]

    
    for k in avg_keys:
        if k == 'SODA_m': 
            continue
        else:
            values = [v.get(k, 0.0) for v in per_video_results]
        final_result[k] = float(sum(values)) / max(len(values), 1)

    final_result["SODA_m"] = process_soda(per_video_results)
    final_result["num_videos"] = num_videos
    final_result["model_name"] = eval_model_name

    # ---------------------------------------
    # ✅ Save logs to results/{model_name}/log
    # ---------------------------------------
    if eval_model_name is not None:
        save_root = f"results_keypoints/{eval_model_name}/log"
        os.makedirs(save_root, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M")
        avg_path = os.path.join(save_root, f"{timestamp}_total_result.json")
        per_case_path = os.path.join(save_root, f"{timestamp}_per_case_result.json")

        # 保存平均结果
        final_result["valid_gt_video_num"] = valid_gt_vnum
        final_result["valid_pred_video_num"] = valid_pred_vnum
        with open(avg_path, "w", encoding="utf-8") as fw:
            json.dump(final_result, fw, indent=4, ensure_ascii=False)

        # 保存每个视频 case 结果
        with open(per_case_path, "w", encoding="utf-8") as fw:
            json.dump(per_video_results, fw, indent=4, ensure_ascii=False)

        
        print(f"✅ Saved corpus avg result to: {avg_path}")
        print(f"✅ Saved per case result to:   {per_case_path}")

    return final_result




def evaluate_single_dense_captions(
    predicted_segments,
    gt_segments,
    predicted_captions,
    gt_captions,
    key="this_video", # video id for this <gt, pred> pairs
    scorers=None,
    split=None,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9),
    soda=True,
    tmponly=False,
    debug=False,
):
    """Compute both the P/R and NLP metrics for the given predictions.

    Args:
     predicted_segments: A numpy arrays, of shape [K x 2]
         containing the predicted segment intervals.
     gt_segments: A numpy arrays, of shape [S x 2]
         containing the ground truth segment intervals.
     predicted_captions: A list, of string of shape [K]
         containing the corresponding K predicted captions.
     gt_captions: A list, of strings of shape [S] containing the
         corresponding S ground truth captions.
     splits: A numpy array, of shape [S] indicating
         the annotation set (1/2 for ActivityNet).
     key: A string of video id
     iou_thresholds: A list of thresholds for IOU to average over.
     soda: Whether to compute SODA or not.
     tmponly: In this case do not compute captioning metrics.
     scorers: caption metric mapping strings to scorers.

    Returns:
        (precision, recall): The precision and recall of the detections averaged
        over the IOU thresholds.
        metrics: The NLP metrics of the predictions averaged over the IOU thresholds.
    """
    if scorers is None:
        scorers = {
                'Checklist_Score': Checklist_Score(),
                # 'METEOR': Meteor(),
        }
    # ------------------------------------------------
    # ✅ Step 1. 计算定位指标（Precision / Recall / F1）
    # ------------------------------------------------
    # splits 用于兼容 ActivityNet，多 GT 可来自不同 annotator
    # 如果不提供 split，我们自己创建一个
    if split is None:
        split = np.zeros(len(gt_segments))  # 全部属于同一个 split


    if debug:
        ipdb.set_trace()
    detection_precision, detection_recall, iou_matrices = evaluate_detections(
        predicted_segments, gt_segments, split, iou_thresholds
    )
    
    metric_tiou = {}
    mean_precision = sum(detection_precision) / len(detection_precision)
    mean_recall = sum(detection_recall) / len(detection_recall)
    for j, threshold in enumerate(iou_thresholds):
            metric_tiou[f'Precision@{threshold}'] = float(detection_precision[j])
            metric_tiou[f'Recall@{threshold}'] = float(detection_recall[j])
    metric_tiou['Precision_Mean'] = float(mean_precision)
    metric_tiou['Recall_Mean'] = float(mean_recall)
    metric_tiou['F1_Score'] = 2 * float(mean_recall) * float(mean_precision) / (float(mean_recall) + float(mean_precision)) if float(mean_recall) + float(mean_precision) > 0 else 0.0
    
    if debug:
        print("Finished evaluating detection metrics F1:")
        # print("iou_matrices:", iou_matrices)
        print("F1 score:", metric_tiou)
        
    result = metric_tiou
    result["n_preds"] = len(predicted_captions)
    result["key"] = key

    # 如果只算定位指标，直接返回F1, 不用再计算soda_m
    if tmponly:
        return result
    
    # -------------- Step 2: Temporal-aligned Caption-level Evaluation (SODA_M) --------------
    # SODA_M (Merged) is the improved version of SODA_c that merged multiple predicted captions for each GT caption.

    #
    if soda:
        fs = soda_m(iou_matrices, scorers['Checklist_Score'],
                    predicted_captions, gt_captions, split, 0.0, video_id=key, debug=debug, predicted_segments=predicted_segments, gt_segments=gt_segments)  
        result["SODA_m"] = fs     # <-- 只返回 SODA_m
        if debug:
            print("==> Final Soda_m Score:")
            print(result["SODA_m"])
            ipdb.set_trace()

    return result

def _normalize_pred_item_list(item_list):
    """
    将 [ {timestamp: [...或"mm:ss-mm:ss"], caption: "..."} , ... ]
       或 [ {timestamp: "...", segment_detail_caption*/video_background*/...} , ... ]
    统一规整为:
        [ {"timestamp":[s,e], "caption": str}, ... ]
    """

    norm = []

    for it in (item_list or []):
        if not isinstance(it, dict):
            continue

        # ---- timestamp ----
        ts = it.get("timestamp")
        if ts is None:
            continue
        try:
            ts = unify_timestamp_format(seg["timestamp"])
        except Exception as e: # 使用 Exception 避免捕获 KeyboardInterrupt 等系统信号
            logging.warning(f"Failed to normalize timestamp: {seg.get('timestamp')}, Error: {e}")
            continue
        # ---- caption ----
        # ① 第一种：有 caption，直接用
        if isinstance(it.get("caption"), str) and it["caption"].strip():
            caption_text = it["caption"].strip()

        else:
            # ② 第二种：没有 caption → 直接调用 merge_fields()
            # merge_fields() 会：自动探测 *_en 或无后缀字段，拼接成 caption
            caption_text, _ = merge_fields(it)

        norm.append({"timestamp": ts, "caption": caption_text})

    return norm


def parse_sent(sent):
        """Sentence preprocessor."""
        res = re.sub('[^a-zA-Z]', ' ', sent)
        res = res.strip().lower().split()
        return res


def unify_timestamp_format(ts):
    """
    统一 timestamp 为 [float start_sec, float end_sec]
    支持格式：
        - [start, end]
        - "mm:ss-mm:ss"
        - "ss-ss"
        - "mm:ss-ss" 或 "ss-mm:ss" （混合）
        - 自动去除空格和前导零
    """
    def to_seconds(x: str) -> float:
        """辅助函数：'mm:ss' 或 'ss' -> 秒"""
        x = x.strip()
        if ":" in x:
            parts = x.split(":")
            if len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            elif len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
            else:
                raise ValueError(f"Invalid time string: {x}")
        else:
            return float(x)

    # ✅ case 1: list or tuple
    if isinstance(ts, (list, tuple)) and len(ts) == 2:
        return [float(ts[0]), float(ts[1])]

    # ✅ case 2: string like "00:01-00:23" / "18-00:25"
    if isinstance(ts, str):
        ts = ts.strip()
        if "-" not in ts:
            raise ValueError(f"Invalid timestamp format (no '-'): {ts}")
        s, e = ts.split("-", 1)
        try:
            start = to_seconds(s)
            end = to_seconds(e)
            return [start, end]
        except Exception:
            raise ValueError(f"Invalid timestamp format: {ts}")

    raise ValueError(f"Timestamp format not recognized: {ts}")


def merge_fields(ann):
    """把字段拼接成 caption string；自动 fallback *_en → 无 _en"""

    fields = [
        "segment_detail_caption_en",
        "video_background_en",
        "acoustics_content_en",
        "shooting_style_en",
        "speech_content_en",
        "camera_state_en",
        "storyline_en",
        "keypoints"
    ]

    merged = []
    flt_dict = {}

    for f in fields:
        candidates = [f, f[:-3]]  # *_en → 无后缀 fallback

        for key in candidates:
            if key in ann and isinstance(ann[key], str) and ann[key].strip():
                merged.append(f"[{key}]\t" + ann[key].strip())
                flt_dict[key] = ann[key].strip()
                break  # ✅ 最先命中的一个用掉即可
            elif key in ann and isinstance(ann[key], dict):
                # get gt keypoints
                flt_dict[key] = ann[key]

    merged_anno = "\n".join(merged).strip()
    if merged_anno:
        merged_anno += "."

    return merged_anno, flt_dict




def _normalize_from_gt_like_ann(seg_list):
    """
    将 GT-like 的分段标注 (human_json_anno / human_json_anno_en) 规整为预测格式
    返回: [ {"timestamp":[s,e], "caption": merged_text}, ... ]
    """
    out = []
    for seg in (seg_list or []):
        if "timestamp" not in seg:
            continue
        try:
            ts = unify_timestamp_format(seg["timestamp"])
        except Exception as e: # 使用 Exception 避免捕获 KeyboardInterrupt 等系统信号
            logging.warning(f"Failed to normalize timestamp: {seg.get('timestamp')}, Error: {e}")
            continue
        # 若原条目已有 caption 字段则优先；否则使用 merge_fields(seg) 生成一个合并文本
        if "caption" in seg and isinstance(seg["caption"], str) and seg["caption"].strip():
            caption_text = seg["caption"].strip()
        else:
            merged_text, _flt = merge_fields(seg)  # 你已有的方法，合并 6~7 维为一句话
            caption_text = merged_text
        out.append({"timestamp": ts, "caption": caption_text})
    return out



def str_to_json(json_str):
    if not json_str:
        return ""

    # 1. 预处理：移除 Markdown 标记和可能的 API 错误信息
    json_str = json_str.strip()
    if json_str.startswith("```json"): json_str = json_str[7:]
    if json_str.startswith("```"): json_str = json_str[3:]
    if json_str.endswith("```"): json_str = json_str[:-3]
    
    # 如果开头不是 [ 或 {，说明可能包含了 "FAILED" 或其他杂质，尝试找到第一个 [
    if not json_str.startswith("[") and not json_str.startswith("{"):
        start_idx = json_str.find("[")
        if start_idx != -1:
            json_str = json_str[start_idx:]
        else:
            # 彻底无法挽救
            return ""

    # 2. 定义我们预期的所有字段名 (根据你的日志补充完整)
    # 顺序不重要，重要的是要包含所有可能出现的 Key
    known_keys = [
        "timestamp", "segment_detail_caption", "camera_state", 
        "video_background", "storyline", "shooting_style", 
        "speech_content", "acoustics_content", "segment_id",
        "character_list", "visual_content" # 根据需要补充
    ]
    
    # 构建一个匹配 "Key": " 的正则，用来定位字段开始
    # 解释：匹配 "key": " (允许冒号周围有空格)
    keys_pattern = '|'.join(known_keys)
    # 这里的正则意思是：寻找 "key": " 这样的结构
    pattern = re.compile(r'"(' + keys_pattern + r')"\s*:\s*"')

    # 3. 扫描所有字段头的位置
    matches = list(pattern.finditer(json_str))
    
    if not matches:
        # 如果连一个 Key 都没找到，说明数据严重损坏
        return None

    # 4. 重构字符串
    # 我们将把字符串切分成片段，只处理片段中间的内容
    new_parts = []
    
    # 处理第一个 Key 之前的部分 (通常是 [{" )
    new_parts.append(json_str[:matches[0].end()])

    for i in range(len(matches)):
        current_match = matches[i]
        start_content = current_match.end()
        
        # 确定当前字段内容的结束位置
        if i < len(matches) - 1:
            # 如果还有下一个 Key，那么内容结束于下一个 Key 的开始之前
            # 我们往回找，找到上一个字段结束的引号和逗号
            next_match = matches[i+1]
            end_content = next_match.start()
            
            # 截取原始内容
            raw_content_chunk = json_str[start_content:end_content]
            
            # 核心修复逻辑：
            # 这个 chunk 应该以 ", 结尾（或者换行空白符 + ",）
            # 我们从右往左找最后一个双引号，认为它是闭合引号
            last_quote_idx = raw_content_chunk.rfind('"')
            
            if last_quote_idx != -1:
                # 内容是：从开始 到 最后一个引号之前
                content_body = raw_content_chunk[:last_quote_idx]
                # 结尾符号是：最后一个引号及其后面的逗号/空白
                suffix = raw_content_chunk[last_quote_idx:]
                
                # 转义 content_body 里的所有引号
                fixed_body = content_body.replace('"', '\\"')
                # 重新拼接
                new_parts.append(fixed_body + suffix)
            else:
                # 这种情况很少见，说明结构乱了，直接拼回去
                new_parts.append(raw_content_chunk)
            
            # 把下一个 Key 的头也拼进去
            new_parts.append(json_str[next_match.start():next_match.end()])
            
        else:
            # 这是最后一个 Key (通常是 acoustics_content 或 speech_content)
            # 这也是最容易发生截断（Truncation）的地方
            raw_content_chunk = json_str[start_content:]
            
            # 检查是否截断
            # 正常的结尾应该是 "}] 或 "}]... 等
            # 我们尝试找最后一个引号
            last_quote_idx = raw_content_chunk.rfind('"')
            
            # 判断是否是合法的结尾结构 (检查最后几个非空字符)
            stripped_chunk = raw_content_chunk.strip()
            is_valid_end = stripped_chunk.endswith('}]') or stripped_chunk.endswith('}') or stripped_chunk.endswith(']')
            
            if is_valid_end and last_quote_idx != -1:
                # 看起来是完整的
                content_body = raw_content_chunk[:last_quote_idx]
                suffix = raw_content_chunk[last_quote_idx:]
                fixed_body = content_body.replace('"', '\\"')
                new_parts.append(fixed_body + suffix)
            else:
                # 看起来是截断的 (Unterminated string)
                # 既然是截断，剩下的所有内容都属于这个字段的值
                # 我们要把里面所有的引号都转义，然后强制闭合
                fixed_body = raw_content_chunk.replace('"', '\\"')
                new_parts.append(fixed_body)
                
                # 强制补全 JSON 结构
                new_parts.append('"}]') 

    # 5. 拼接结果
    repaired_str = "".join(new_parts)

    # 6. 最后的尝试解析
    try:
        return json.loads(repaired_str)
    except json.JSONDecodeError:
        # 如果依然失败，尝试更激进的清理（处理控制字符等）
        try:
            repaired_str = repaired_str.replace("\n", "\\n")
            return json.loads(repaired_str)
        except:
            return ""

def load_pred_file(pred_file_path):
    """
    支持四种固定格式，统一返回:
    pred_data: dict[str, list[{"timestamp":[s,e], "caption": str}]]
               key 为 clip_path（尽量保持原样）；若没有 clip_path 则跳过
    """
    pred_data = {}

    # 先判断是否是 .jsonl
    is_jsonl = pred_file_path.lower().endswith(".jsonl")

    with open(pred_file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # ipdb.set_trace()
    # 简单启发：如果是 .jsonl 或者内容中出现了多行 JSON 对象，就按 jsonl 处理
    if is_jsonl or ("\n{" in raw.strip().splitlines()[0] and "\n" in raw):
        # ---------- 处理 jsonl ---------
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # 【格式4】一行一个：{"clip_path": "...", "pred": [ {...}, ... ]}
            if isinstance(obj, dict) and "clip_path" in obj and "pred" in obj:
                clip_path = obj["clip_path"]
                item_list = _normalize_pred_item_list(obj.get("pred", []))
                if clip_path:
                    pred_data.setdefault(clip_path, []).extend(item_list)
                continue

            # 【格式2】一行一个：与 GT 一模一样的结构（human_json_anno / human_json_anno_en）
            if isinstance(obj, dict) and "clip_path" in obj and (
                "prediction_json" in obj or "prediction" in obj
            ):
                clip_path = obj["clip_path"]
                segs = obj.get("prediction_json") or obj.get("prediction_json_en") or obj.get("prediction")
                if isinstance(segs, str):
                    segs = str_to_json(segs)

                item_list = _normalize_from_gt_like_ann(segs)
                if clip_path:
                    pred_data.setdefault(clip_path, []).extend(item_list)
                continue

            # 其它 jsonl 行：忽略或后续扩展
        return pred_data

    # ---------- 处理 json ----------
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Cannot parse JSON: {pred_file_path} ({e})")

    # 【格式1】JSON 字典：{ "clip_path": [ {timestamp, caption}, ... ], ... }
    if isinstance(data, dict):
        # 逐个 key 规整
        for clip_path, item_list in data.items():
            if not isinstance(item_list, list):
                continue
            pred_data[clip_path] = _normalize_pred_item_list(item_list)
        return pred_data

    # 【格式3】JSON 列表： [ {"clip_path": "...", "pred": [ {...}, ... ]}, ... ]
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            clip_path = item.get("clip_path")
            if not clip_path:
                continue

            # 可能是 {"clip_path":..., "pred":[...]}（标准第三种）
            if "pred" in item and isinstance(item["pred"], list):
                pred_data.setdefault(clip_path, []).extend(
                    _normalize_pred_item_list(item["pred"])
                )
                continue

            # 也可能是 GT-like 的条目（例如被直接拷到 json 数组里）
            if "human_json_anno_en" in item or "human_json_anno" in item:
                segs = item.get("human_json_anno_en") or item.get("human_json_anno") or []
                pred_data.setdefault(clip_path, []).extend(
                    _normalize_from_gt_like_ann(segs)
                )
                continue

        return pred_data

    # 其它未知结构
    raise ValueError(f"Unsupported pred_file structure in: {pred_file_path}")


def eval_with_files(pred_file, gt_file, split="test", tmponly=False, max_workers=8, debug=False):
    """Evaluate a file of predictions, compatible with both old dict-format
       and new list-format prediction json."""

    smap = {'train': 1, 'val': 2, 'test': 3}

        
    # 实际调用：兼容多种 pred_data json结构
    pred_data = load_pred_file(pred_file)


    # ===== read gt file (jsonl) =====
    with open(gt_file, "r") as f:
        lines = f.readlines()
        gt_data = [json.loads(line) for line in lines]

    print(f"[{split}] gt video nums {len(gt_data)}; pred video nums {len(pred_data)}")

    split = smap[split]

    # -----------------------------
    #     preprocess GT data
    # -----------------------------
    keys = []
    gt_segments = []
    gt_captions = []
    splits = []

    for jterm in gt_data:
        keys.append(jterm["clip_path"])   # <-- GT 使用 clip_path 对齐
        segs = jterm["human_json_anno"] if "human_json_anno" in jterm else jterm["human_json_anno_en"]

        segments = []
        captions = []
        for seg in segs:
            # ts = unify_timestamp_format(seg["timestamp"])  # timestamp -> [start,end]
            try:
                ts = unify_timestamp_format(seg["timestamp"])
            except Exception as e: # 使用 Exception 避免捕获 KeyboardInterrupt 等系统信号
                logging.warning(f"Failed to normalize timestamp: {seg.get('timestamp')}, Error: {e}")
                continue
            segments.append(ts)

            merged_cap, cap_dict = merge_fields(seg)
            captions.append(cap_dict)

        gt_segments.append(np.array(segments))
        gt_captions.append(np.array(captions))
        splits.append(np.array([split] * len(segments)))

    # -----------------------------
    #     preprocess PRED data
    # -----------------------------
    predicted_segments = []
    predicted_captions = []
    predicted_paras = []

    aligned_keys = []
    aligned_splits = []
    aligned_gt_segments = []
    aligned_gt_captions = []

    pred_video_count = 0

    for i, vid in enumerate(keys):

        vid_basename = os.path.basename(vid)

        # 支持：完整 clip_path 和 basename 形式
        if vid in pred_data:
            pred_key = vid
        elif vid_basename in pred_data:
            pred_key = vid_basename
        else:
            pred_key = None

        aligned_keys.append(vid)
        aligned_gt_segments.append(gt_segments[i])
        aligned_gt_captions.append(gt_captions[i])
        aligned_splits.append(splits[i])

        if pred_key is None:
            predicted_segments.append(np.array([]))
            predicted_captions.append(np.array([]))
            predicted_paras.append("")
            continue

        pred_video_count += 1
        video_pred_list = pred_data[pred_key]

        one_segments = []
        one_captions = []

        for pred_item in video_pred_list:
            # ts = unify_timestamp_format(pred_item["timestamp"])
            try:
                ts = unify_timestamp_format(seg["timestamp"])
            except Exception as e: # 使用 Exception 避免捕获 KeyboardInterrupt 等系统信号
                logging.warning(f"Failed to normalize timestamp: {seg.get('timestamp')}, Error: {e}")
                continue
            one_segments.append(ts)
            one_captions.append(pred_item.get("caption", ""))

        predicted_segments.append(np.array(one_segments))
        predicted_captions.append(np.array(one_captions))
        predicted_paras.append("\n".join(one_captions))

    print(f"GT 视频数量: {len(gt_data)}, 有预测的视频数量: {pred_video_count}")
    print(f"最终评测采样视频数量: {len(aligned_keys)}")

    # 更新变量
    keys = aligned_keys
    splits = aligned_splits
    gt_segments = aligned_gt_segments
    gt_captions = aligned_gt_captions

    print('Evaluate dense video captioning ...')
    assert len(predicted_segments) == len(gt_segments) == len(predicted_captions), \
        "predicted and gt data length mismatch!"

    pred_model = pred_file.split("/")[-1].split(".")[0]
    valid_gt_vnum = len(gt_data)
    valid_pred_vnum = pred_video_count
    if debug: # 单线程评测，方便跟踪
        results = evaluate_dense_captions(
            valid_gt_vnum, valid_pred_vnum, predicted_segments, gt_segments, predicted_captions, gt_captions,
            keys, tmponly=tmponly, max_workers=max_workers, soda=True, eval_model_name=pred_model, debug=debug
        )
    else: # 多线程评测
        results = evaluate_dense_captions_para(
            valid_gt_vnum, valid_pred_vnum, predicted_segments, gt_segments, predicted_captions, gt_captions,
            keys, tmponly=tmponly, max_workers=max_workers, soda=True, eval_model_name=pred_model
        )
    print("==>Final evaluation results:")
    print(results)
    print()

    return results





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default='/home/gaohuan03/yaolinli/code/qwen25omni/OmniVideoCaption/Eval/t-score/inference_files/v4_18k_epoch1_from_qwen2.5omni.jsonl')
    parser.add_argument('--gt_file', type=str, default='/home/gaohuan03/yaolinli/code/qwen25omni/OmniVideoCaption/Eval/t-score/ours_gt_file_keypoints.json')
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--tmponly', action="store_true", default=False)
    parser.add_argument('--analyze', action="store_true", default=False)
#     parser.add_argument('--paragraph', action="store_true", default=False)
    args = parser.parse_args()
    
    DEBUG_MODE = True if args.debug else False
    results = eval_with_files(args.pred_file, args.gt_file, tmponly=args.tmponly, max_workers=args.max_workers, debug=DEBUG_MODE)
    
    # save sorted metric scores for evaluated cases
    if args.analyze:
        # keys for results:
        # ['CIDER', 'METEOR', 'Precision@0.3', 'Recall@0.3', 'Precision@0.5', 'Recall@0.5', 'Precision@0.7', 'Recall@0.7', 'Precision@0.9', 'Recall@0.9', 'Precision_Mean', 'Recall_Mean', 'F1_Score', 'SODA_c_2', 'n_preds', 'key', 'SODA_c_1']
        output_path = '/'.join(args.pred_file.split("/")[:-1])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = "score_" + args.pred_file.split("/")[-1]
        output_file = os.path.join(output_path, output_file)
        vids = results["key"]
        CIDEr = results["CIDER"]
        n_preds = results["n_preds"]
        P = results["Precision_Mean"]
        R = results["Recall_Mean"]
        f1_score = results["F1_Score"]
        # sort f1_score and return indexs from max to min
        idxs = np.argsort(f1_score)[::-1]
        good_cases = []
        for idx in idxs:
            good_cases.append(vids[idx] + "\tPrecision:" + str(round(P[idx]*100, 1)) + "\tRecall:" + str(round(R[idx]*100, 1)) + "\tF1_score:" + str(round(f1_score[idx]*100, 1)) + "\tCider: " + str(round(CIDEr[idx]*100, 1)) + "\tn_preds: " + str(round(n_preds[idx], 1)))
        with open(output_file, "w") as f:
            f.write("\n".join(good_cases))
        print(f"save sorted metrics for cases at \n {output_file}")
