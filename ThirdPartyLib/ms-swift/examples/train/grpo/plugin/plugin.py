import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional,Any,Tuple
import json
import torch
from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger
import json
import collections
import logging
import random
import re
import string
import argparse
import numpy as np
import pdb
from func_timeout import func_timeout, FunctionTimedOut


logger = get_logger()


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
                cache_dir: str = "cache_checklist_judge",
                model_name: str = "gemini-2.5-flash",
                credentials: str = "/home/gaohuan03/yaolinli/code/qwen25omni/OmniVideoCaption/project_config/mmu-gemini-caption-1-5pro-86ec97219196.json",
                eval_model_name: str = None,
                debug: bool = False):
        self.max_workers = max_workers
        #self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.credentials = credentials
        self._write_lock = Lock()
        self.debug=debug
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
        # logging.basicConfig(
        #     filename=log_file,
        #     level=logging.INFO,
        #     format="%(asctime)s [%(levelname)s] %(message)s"
        # )

        # --- Gemini client ---
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials
        try:
            user_info = json.load(open(self.credentials))
            self.client = genai.Client(vertexai=True, project=user_info["project_id"], location="global")
        except Exception as e:
            print(f"[Gemini Init] fail: {e}")
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

    # yll-update-0107-fix
    # wyc-update-1231
    def _call_gemini_retry(self, prompt: str, retries: int = 4, base_delay: float = 1.2, timeout_sec: int = 180) -> Dict[str, Any]:
        for attempt in range(retries):
            # 提前计算等待时间
            wait = base_delay * (2 ** attempt)
            
            try:
                # 核心修改：使用 func_timeout 替代 ThreadPoolExecutor 来控制超时
                # args=(prompt,) 传递参数给 self._call_gemini_once
                result = func_timeout(timeout_sec, self._call_gemini_once, args=(prompt,))
                
                # --- 严格保留原有的校验逻辑 ---
                # 校验结果格式
                if result and "by_dim" in result:
                    return result
                else:
                    raise ValueError(f"Judge JSON missing 'by_dim'. Got: {str(result)[:100]}")

            except FunctionTimedOut:
                # 对应原代码的 concurrent.futures.TimeoutError
                # 发生超时，func_timeout 会自动中断执行，无需手动 shutdown
                print(f"[Gemini Judge] attempt={attempt+1}/{retries} timeout (> {timeout_sec}s);")

            except Exception as e:
                # 对应原代码的 Exception 捕获
                print(f"[Gemini Judge] attempt={attempt+1}/{retries} failed: {e}; sleep {wait:.1f}s")

            # 执行退避等待
            time.sleep(wait)

        # --- 严格保留原有的失败保底返回 ---
        print("[Gemini Judge] all retries failed.")
        empty = {
            "gt_count": 0,
            "gt_keypoints": [],
            "correct_count": 0,
            "correct_keypoints": []
        }
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
            if isinstance(raw_gt, dict):
                gt_norm = {dim: raw_gt.get(dim, {}).get("gt_keypoints", [])
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


            # 训练时不写入缓存
            # ✅ 读取缓存 / 调用 Gemini 
            # if cpath.exists():
            #     out = json.load(open(cpath, "r", encoding="utf-8"))
            # else:
            prompt = self._build_prompt(gt_norm, pred_caption)
            out = self._call_gemini_retry(prompt)

            # ✅ 保存用于排查问题
            out["sample_tag"] = sample_tag
            out["gt_keypoints"] = gt_norm
            out["pred_caption"] = pred_caption

            # with self._write_lock:
            #     with open(cpath, "w", encoding="utf-8") as fw:
            #         json.dump(out, fw, ensure_ascii=False)
            # ============= 关键更改：使用 judge 返回的 gt_count / correct_count =============
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
                    print(f"[compute_score aggregation] {e}")
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
            if debug:
                print("===>dim scores")
                print(dim_scores)
                print()

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
    return fs, merged_groups

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
    # 如果预测片段为空，说明模型没有输出任何有效的时间段
    if len(predicted_segments) == 0:
        # 返回全 0 的结果。
        # 注意：这里返回的字典结构必须与你代码后续逻辑期望的结构一致。
        # 通常 t-score 或 dense caption evaluation 返回的是类似以下的结构：
        empty_result = {
            'F1_Score': 0.0,
            'mIoU': 0.0,
            'SODA_m': None,
        }
        return empty_result
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


    _, _, raw_iou_matrices = evaluate_detections(
        predicted_segments, gt_segments, split, iou_thresholds
    )
    current_split_id = int(split[0]) if len(split) > 0 else 0
    
    # 初始化变量
    merged_groups = [] 
    soda_score = None
    
    # -------------------------------------------------------------------------
    # Step 2. 获取 Merged Groups (Branching Logic)
    # -------------------------------------------------------------------------
    # 如果不是仅计算临时指标，且开启了 SODA，则调用 soda_m 并获取 groups
    if (not tmponly) and soda:
        # 假设 soda_m 已经修改为返回 (score, merged_groups)
        # 注意：你需要确保 soda_m 接受 raw_iou_matrices 作为输入
        soda_score, merged_groups = soda_m(
            raw_iou_matrices, 
            scorers['Checklist_Score'],
            predicted_captions, 
            gt_captions, 
            split, 
            0.0, 
            video_id=key, 
            debug=debug, 
            predicted_segments=predicted_segments, 
            gt_segments=gt_segments
        )
        if debug:
            print(f"Video {key}: Retrieved merged_groups from SODA_m calculation.")

    else:
        # 如果是 tmponly 或者 soda=False，我们需要手动计算 merged_groups
        if current_split_id in raw_iou_matrices:
            iou_matrix = raw_iou_matrices[current_split_id]
            # 2.1 DP Assignment
            _, pairs = chased_dp_assignment(iou_matrix)
            # 2.2 Horizontal Expand
            merged_groups = horizontal_expand_pairs(iou_matrix, pairs)
            
            if debug:
                print(f"Video {key}: Calculated merged_groups manually (tmponly={tmponly}).")

    # -------------------------------------------------------------------------
    # Step 3. 重构预测片段 (Reconstruct Segments based on merged_groups)
    # -------------------------------------------------------------------------
    new_pred_list = []
    covered_pred_indices = set()
    
    # 3.1 添加合并后的片段 (Merged Segments)
    # merged_groups 结构通常为: [(gt_idx, [pred_idx1, pred_idx2, ...]), ...]
    for gt_idx, pred_indices in merged_groups:
        try:
            if len(pred_indices) == 0: continue
            
            group_segments = predicted_segments[pred_indices]
            new_start = np.min(group_segments[:, 0])
            new_end = np.max(group_segments[:, 1])
            
            new_pred_list.append([new_start, new_end])
            covered_pred_indices.update(pred_indices)
        except Exception as e:
            # 【修改点】不要使用 ipdb，改为打印错误日志
            if debug:
                print(f"[Warning] Video {key}: Error merging segments for gt_idx {gt_idx}.")
                print(f"Error info: {e}")
                print(f"pred_indices: {pred_indices}")
            continue
    # 3.2 添加未被合并的孤立片段 (Orphan Segments / False Positives)
    num_orig_preds = predicted_segments.shape[0]
    for i in range(num_orig_preds):
        if i not in covered_pred_indices:
            new_pred_list.append(predicted_segments[i])
    
    if len(new_pred_list) > 0:
        merged_predicted_segments = np.array(new_pred_list)
    else:
        merged_predicted_segments = np.zeros((0, 2))

    # -------------------------------------------------------------------------
    # Step 4. 计算最终定位指标 (Precision / Recall / F1 / mIoU)
    # -------------------------------------------------------------------------
    # 我们需要重新计算基于合并后片段的 IoU 矩阵
    # evaluate_detections 返回: precision, recall, iou_matrices
    detection_precision, detection_recall, merged_iou_matrices = evaluate_detections(
        merged_predicted_segments, gt_segments, split, iou_thresholds
    )
    
    # --- 计算 mIoU ---
    # 逻辑：对于每个 GT，找到与其 IoU 最大的 Pred，取该 IoU 值，然后对所有 GT 求平均。
    # 这种计算方式通常被称为 "Average Best IoU" 或 "mIoU at GT level"。
    # 另一种方式是对所有 IoU > 0 的匹配对求平均，但前者在检测任务中更常用。
    
    miou = 0.0
    if current_split_id in merged_iou_matrices:
        iou_mat = merged_iou_matrices[current_split_id] # shape: [num_preds, num_gts]
        if iou_mat.size > 0:
            # 1. 沿 axis=0 (preds) 取最大值，得到每个 GT 对应的最大 IoU
            # shape: [num_gts]
            max_ious_per_gt = np.max(iou_mat, axis=0)
            
            # 2. 计算平均值
            if len(max_ious_per_gt) > 0:
                miou = float(np.mean(max_ious_per_gt))
    # -----------------

    metric_tiou = {}
    if len(detection_precision) > 0:
        mean_precision = sum(detection_precision) / len(detection_precision)
        mean_recall = sum(detection_recall) / len(detection_recall)
    else:
        mean_precision = 0.0
        mean_recall = 0.0
    for j, threshold in enumerate(iou_thresholds):
            metric_tiou[f'Precision@{threshold}'] = float(detection_precision[j])
            metric_tiou[f'Recall@{threshold}'] = float(detection_recall[j])
    metric_tiou['Precision_Mean'] = float(mean_precision)
    metric_tiou['Recall_Mean'] = float(mean_recall)
    denom = float(mean_recall) + float(mean_precision)
    metric_tiou['F1_Score'] = 2 * float(mean_recall) * float(mean_precision) / denom if denom > 0 else 0.0
    # 添加 mIoU 到结果字典
    metric_tiou['mIoU'] = miou
    
    result = metric_tiou
    result["n_preds"] = len(predicted_captions)
    result["key"] = key

    # 如果计算了 SODA 分数，加入结果中
    if soda_score is not None:
        result["SODA_m"] = soda_score
        if debug:
            print("==> Final Soda_m Score:", result["SODA_m"])
            print("==> Final Detection Metrics:", metric_tiou)
            # ipdb.set_trace()

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
        ts = unify_timestamp_format(ts)      # 兼容 [s,e] / "mm:ss-mm:ss"

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


# def unify_timestamp_format(ts):
#     """
#     统一 timestamp 为 [float start_sec, float end_sec]
#     支持格式：
#         - [start, end]
#         - "mm:ss-mm:ss"
#         - "ss-ss"
#         - "mm:ss-ss" 或 "ss-mm:ss" （混合）
#         - 自动去除空格和前导零
#     """
#     def to_seconds(x: str) -> float:
#         """辅助函数：'mm:ss' 或 'ss' -> 秒"""
#         x = x.strip()
#         if ":" in x:
#             parts = x.split(":")
#             if len(parts) == 2:
#                 m, s = parts
#                 return float(m) * 60 + float(s)
#             elif len(parts) == 3:
#                 h, m, s = parts
#                 return float(h) * 3600 + float(m) * 60 + float(s)
#             else:
#                 raise ValueError(f"Invalid time string: {x}")
#         else:
#             return float(x)

#     # ✅ case 1: list or tuple
#     if isinstance(ts, (list, tuple)) and len(ts) == 2:
#         return [float(ts[0]), float(ts[1])]

#     # ✅ case 2: string like "00:01-00:23" / "18-00:25"
#     if isinstance(ts, str):
#         ts = ts.strip()
#         if "-" not in ts:
#             raise ValueError(f"Invalid timestamp format (no '-'): {ts}")
#         s, e = ts.split("-", 1)
#         try:
#             start = to_seconds(s)
#             end = to_seconds(e)
#             return [start, end]
#         except Exception:
#             raise ValueError(f"Invalid timestamp format: {ts}")

#     raise ValueError(f"Timestamp format not recognized: {ts}")


from typing import List, Union, Optional

def unify_timestamp_format(ts: Union[str, list, tuple]) -> Optional[List[float]]:
    """
    统一 timestamp 为 [float start_sec, float end_sec]。
    
    安全设计：
    - 解析成功：返回 [start, end]
    - 解析失败（格式错误、非数字等）：返回 None
    - 绝不抛出异常 (No Raise)
    """
    
    def to_seconds(x: str) -> Optional[float]:
        """辅助函数：安全地将 'mm:ss', 'h:m:s' 或 'ss' 转为秒"""
        try:
            x = str(x).strip() # 强制转str并去空格
            if ":" in x:
                parts = x.split(":")
                if len(parts) == 2: # mm:ss
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3: # hh:mm:ss
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                else:
                    return None # 格式怪异，如 12:34:56:78
            else:
                # 纯数字情况
                return float(x)
        except (ValueError, TypeError):
            # 捕获 float转换失败（如 "abc"）或 split 失败
            return None

    # ✅ Case 1: List or Tuple (e.g., [10.5, 20.0])
    if isinstance(ts, (list, tuple)):
        if len(ts) != 2:
            return None
        try:
            # 尝试将元素强转为 float
            return [float(ts[0]), float(ts[1])]
        except (ValueError, TypeError):
            return None

    # ✅ Case 2: String (e.g., "00:01-00:23")
    if isinstance(ts, str):
        ts = ts.strip()
        
        # 必须包含分隔符 "-"
        if "-" not in ts:
            return None
            
        # 只分割第一个 "-"，防止出现 "00:01-00:23-extra" 导致的错误逻辑
        parts = ts.split("-", 1)
        
        start = to_seconds(parts[0])
        end = to_seconds(parts[1])
        
        # 只要有一个解析失败，整体就失败
        if start is None or end is None:
            return None
            
        return [start, end]

    # 其他未识别类型
    return None

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


# def check_timestamp_list_format(completion: str) -> Tuple[bool, List[Dict[str, Any]]]:
#     try:
#         completion_list = json.loads(completion)
#     except Exception as e:
#         print("completion could not be parsed as timestamp_list:", e)
#         return False, []

#     if isinstance(completion_list, list) and len(completion_list) != 0:
#         return True, completion_list
#     else:
#         return False, []

class DummyReward(ORM):
    def __init__(self):
        pass

    def __call__(self,completions,**kwargs)->List[float]:
        import random
        rewards=[]
        for i in completions:
            rewards.append(random.random())
        return rewards


###工具函数
def parse_timestamp(time_str: str) -> float:
    """将时间字符串转换为秒数"""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def check_timestamp_list_format(json_str: str) -> Tuple[bool, List[Dict[str, Any]]]:
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
        return True,json.loads(repaired_str)
    except json.JSONDecodeError:
        # 如果依然失败，尝试更激进的清理（处理控制字符等）
        try:
            repaired_str = repaired_str.replace("\n", "\\n")
            return True,json.loads(repaired_str)
        except:
            return False,[]

def check_dict_format(item:Dict[str,Any])->bool:
    required_keys = [
        "timestamp", "segment_detail_caption", "camera_state", "video_background",
        "storyline", "shooting_style", "speech_content", "acoustics_content",
    ]
    if not isinstance(item, dict):
            return False
    for k in required_keys:
        if k not in item or not isinstance(item[k], str):
            return False
    return True

class TimestampFormat(ORM):
    def __init__(self):
        pass

    def __call__(self,completions:List[str],**kwargs)->List[float]:
        rewards=[]
        print("\n正在计算Timestamp Format rewards:\n")

        for content in completions:
            try:
                reward=0.0
                timestamp_format_flag,timestamp_list=check_timestamp_list_format(content)

                if timestamp_format_flag:
                    for seg in timestamp_list:
                        if check_dict_format(seg):
                            reward+=1/len(timestamp_list)
                else:
                    reward=0.0

                print(f"\ncompletion:{content} \n TimestampFormat reward:{reward}\n")

            except Exception as e:
                print(f"计算TimestampFormat时触发异常:{e},做保底方案,Reward=0.0,")
                reward=0.0


            rewards.append(reward)
            
        return rewards


class TiemstampCaptionLength(ORM):
    def __init__(self):
        from transformers import Qwen2_5OmniProcessor
        self.qwen_processor=Qwen2_5OmniProcessor.from_pretrained("/home/gaohuan03/yaolinli/code/qwen25omni/weiyuancheng-test/MultiShotCaption/our_code/checkpoint/1016_sft_movie101-mmtrail-qwen2.5omni-ckpt/v0-20251016-220040/checkpoint-706")
        logger.info("TiemstampCaptionLength函数成功加载qwen25o的processor")
        
    def __call__(self,completions:List[str],**kwargs)->List[float]:
        def cal_token_length(prompt,processor):
            inputs = processor(text=prompt, return_tensors="pt", padding=True)
            seq = inputs["input_ids"][0]
            return len(seq)

        print("\n正在计算TimestampCaptionLength rewards:\n")
        rewards=[]
        for completion in completions:
            try:
                reward = 0.0
                ###计算分段长短reward:
                try:
                    timestamp_format_flag,timestamp_list=check_timestamp_list_format(completion)
                    if not timestamp_format_flag:
                        print("\n\ncompletion:{completion}\n timestamp_format_flag为false")
                        reward=0.0
                    else:
                        for seg in timestamp_list:
                            seg_str=json.dumps(seg)
                            if cal_token_length(seg_str,self.qwen_processor)<750: #TODO:取平均值
                                reward+=1/len(timestamp_list)
                            else:
                                reward+=0.0
                except:
                    reward=0.0

                ###计算总长度
                completion_len=cal_token_length(completion,self.qwen_processor)
                if completion_len<4096: #TODO：取平均值
                    reward = 1.0*reward
                    print(f"\ncompletion:{completion} \n 没有发生总体超长 \n TiemstampCaptionLength reward:{reward}\n")
                else:
                    over_ratio = (completion_len - 4096) / 4096
                    penalty = max(0.0, 1 - over_ratio)   # 0~1 之间
                    reward *= penalty
                    print(f"\ncompletion:{completion} \n 发生了总体超长 \n TiemstampCaptionLength reward:{reward}\n")

            except Exception as e:
                print(f"计算TimestampCaptionLength时触发异常:{e},做保底方案,Reward=0.0,")
                reward=0.0

            rewards.append(reward)
        return rewards

###拆开F1和SODA_M
class DenseCaptionF1(ORM):
    def __init__(self):
        ###在这里注册checklist func和gemini api（）
        self.max_workers=16
        self.scorers = {
                'Checklist_Score': Checklist_Score(max_workers=self.max_workers)
            }

    def __call__(self,completions:List[str],solution:List[List[Dict[str,Any]]],**kwargs)->List[float]:
        print("\n正在计算DenseCaptionF1 rewards:\n")
        assert len(completions)==len(solution)

        items_to_process = zip(completions, solution)
        rewards=[]

        for completion,sol in items_to_process:
            try:
                gt_segments = []
                gt_captions = []

                if not isinstance(sol,list):
                    print("sol类型异常:",sol)
                    assert False

                for seg in sol:
                    # --- timestamp 支持 "00:01-00:20" 或 [1.0, 20.0] ---
                    ts = unify_timestamp_format(seg["timestamp"]) # unified as [start_sec, end_sec]

                    gt_segments.append(ts)
                    # --- merge captions into one string ---
                    merged_cap, cap_dict = merge_fields(seg)
                    gt_captions.append(cap_dict)

                reward = 0.0
                ###一切的前提：顺利的解析
                try:
                    timestamp_format_flag,completion_json=check_timestamp_list_format(completion)
                    if not timestamp_format_flag:
                        print(f"\ncompletion:{completion} \n 在计DenseCaptionF1 reward时无法解析为合格的json\n reward:{reward}\n")
                        rewards.append(0.0)
                        continue #
                except Exception as e:
                    print(f"\ncompletion:{completion} \n 在计DenseCaptionF1 reward时调用check_timestamp_list_format出现了error:{e}\n\n reward:{reward}\n")
                    rewards.append(0.0)
                    continue#


                pred_segments=[]
                pred_captions=[]
                for seg in completion_json:
                    if not check_dict_format(seg):
                        print(f"\ncompletion:{completion} \n 在计DenseCaptionF1 reward时\n {seg} dict format不对\n 跳过处理这一段")
                        continue
                    caption_text, caption_dict = merge_fields(seg)
                    ts = unify_timestamp_format(seg["timestamp"])
                    pred_segments.append(ts)
                    pred_captions.append(caption_text)
                
                try:
                    #只计算F1
                    result=evaluate_single_dense_captions(np.array(pred_segments),np.array(gt_segments),np.array(pred_captions),np.array(gt_captions),key=None,scorers=self.scorers,tmponly=True)
                    F1_Score=result["F1_Score"]
                    # SODA_m_total=result["SODA_m"][0]["SODA_m_total"]
                    # reward=F1_Score+SODA_m_total
                    reward=F1_Score
                    print(f"\n\n在计DenseCaptionF1时:\n\n 视频为:{kwargs.get('audio_in_video','未解析到audio')}\n\ncompletion:{completion} \n\n solution:{solution}\n\n F1_Score:{F1_Score} \n\n reward:{reward}")

                except Exception as e:
                    print(f"\n\n\n\n 在计DenseCaptionF1时:\ncompletion:{completion}\n\n solution:{solution}\n\n 调用evaluate_single_dense_captions异常,异常原因为{e}\n\n reward:0.0")
                    reward=0.0
            except Exception as e:
                print(f"计算F1时触发异常:{e},做保底方案,Reward=0.0,")
                reward=0.0

            rewards.append(reward)
        return rewards



class DenseCaptionSodaM(ORM):
    def __init__(self):
        ###在这里注册checklist func和gemini api（）
        self.max_workers=16
        self.scorers = {
                'Checklist_Score': Checklist_Score(max_workers=self.max_workers)
            }

    def __call__(self,completions:List[str],solution:List[List[Dict[str,Any]]],**kwargs)->List[float]:
        print("\n正在计算DenseCaptionSodaM rewards:\n")
        assert len(completions)==len(solution)

        items_to_process = zip(completions, solution)
        rewards=[]

        for completion,sol in items_to_process:
            try:
                gt_segments = []
                gt_captions = []

                if not isinstance(sol,list):
                    print("sol类型异常:",sol)
                    assert False
        
                for seg in sol:
                    # --- timestamp 支持 "00:01-00:20" 或 [1.0, 20.0] ---
                    ts = unify_timestamp_format(seg["timestamp"]) # unified as [start_sec, end_sec]

                    gt_segments.append(ts)
                    # --- merge captions into one string ---
                    merged_cap, cap_dict = merge_fields(seg)
                    gt_captions.append(cap_dict)

                reward = 0.0
                ###一切的前提：顺利的解析
                try:
                    timestamp_format_flag,completion_json=check_timestamp_list_format(completion)
                    if not timestamp_format_flag:
                        print(f"\ncompletion:{completion} \n 在计DenseCaptionSodaM reward时无法解析为合格的json\n reward:{reward}\n")
                        rewards.append(0.0)
                        continue #
                except Exception as e:
                    print(f"\ncompletion:{completion} \n 在计DenseCaptionSodaM reward时调用check_timestamp_list_format出现了error:{e}\n\n reward:{reward}\n")
                    rewards.append(0.0)
                    continue#


                pred_segments=[]
                pred_captions=[]
                for seg in completion_json:
                    if not check_dict_format(seg):
                        print(f"\ncompletion:{completion} \n 在计DenseCaptionSodaM reward时\n {seg} dict format不对\n 跳过处理这一段")
                        continue
                    caption_text, caption_dict = merge_fields(seg)
                    ts = unify_timestamp_format(seg["timestamp"])
                    pred_segments.append(ts)
                    pred_captions.append(caption_text)
                
                try:
                    result=evaluate_single_dense_captions(np.array(pred_segments),np.array(gt_segments),np.array(pred_captions),np.array(gt_captions),key=None,scorers=self.scorers)
                    #F1_Score=result["F1_Score"]
                    try:
                        SODA_m_total=result["SODA_m"][0]["SODA_m_total"]
                    except Exception as e:
                        SODA_m_total=0.0
                    #reward=F1_Score+SODA_m_total
                    reward=SODA_m_total
                    print(f"\n\n在计DenseCaptionSodaM时:\n\n 视频为:{kwargs.get('audio_in_video','未解析到audio')}\n\ncompletion:{completion} \n\n solution:{solution}\n\nSODA_m_total:{SODA_m_total}\n\n reward:{reward}")
                except Exception as e:
                    print(f"\n\n\n\n 在计DenseCaptionSodaM时:\ncompletion:{completion}\n\n solution:{solution}\n\n 调用evaluate_single_dense_captions异常,异常原因为{e}\n\n reward:0.0")
                    reward=0.0

            except Exception as e:
                print(f"计算SodaM时触发异常:{e},做保底方案,Reward=0.0,")
                reward=0.0
            
            rewards.append(reward)
        return rewards
            


###wyc-add
orms['external_dummy_reward']=DummyReward
orms['external_timestamp_format_reward']=TimestampFormat
orms['external_timestamp_length_reward']=TiemstampCaptionLength
orms['external_dense_caption_f1_reward']=DenseCaptionF1
orms['external_dense_caption_sodam_reward']=DenseCaptionSodaM

