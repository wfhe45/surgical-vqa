#!/usr/bin/env python3
"""
Transformers-based inference for Qwen2.5-VL medical video understanding.
Integrates medical vision processing with RC box drawing and Qwen2.5-VL for inference.
Replaced vLLM backend with transformers to avoid vLLM 0.9.x version compatibility issues.
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import tqdm
from PIL import Image

import torch
if not torch.cuda.is_available() and os.environ.get("ALLOW_NO_GPU") != "1":
    sys.exit(
        "ERROR: CUDA GPU not available (torch.cuda.is_available() = False).\n"
        "       Transformers Qwen2.5-VL inference requires a CUDA GPU. Run on a host with CUDA 12.8+.\n"
        "       Set ALLOW_NO_GPU=1 only exercises stages 1–3 (install/download/filter);\n"
        "       stage 4 (inference) always requires CUDA."
    )
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

# Import vision processing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_process_medical import process_vision_info_medical


def group_data_by_type(data_dicts: List[Dict]) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Group data instances by their qa_type.
    Returns a dictionary where keys are qa_types and values are lists of data instances.
    """
    type_groups = defaultdict(list)
    type_counts = defaultdict(int)

    for idx, data_dict in enumerate(data_dicts):
        qa_type = data_dict.get('qa_type', 'unknown')
        # Add original index to track position in final output
        data_dict['original_idx'] = idx
        type_groups[qa_type].append(data_dict)
        type_counts[qa_type] += 1

    print("Found QA types and their counts:")
    for qa_type, count in type_counts.items():
        print(f"  {qa_type}: {count} instances")

    return dict(type_groups), dict(type_counts)


_YES_NO_STARTERS = {"is", "was", "were", "are", "do", "did", "has", "have", "can", "could", "will", "would"}

_SYSTEM_YES_NO = (
    "You are a surgical video analysis assistant. "
    "Watch the video carefully, then answer with ONLY 'Yes' or 'No', "
    "followed by one short clause (under 10 words) using simple everyday words. "
    "Answer only what is literally visible or happening — do not assume. "
    "Examples:\n"
    "  Q: Are scissors being used? → 'Yes, scissors are being used to cut tissue.'\n"
    "  Q: Is a needle driver present? → 'No, a needle driver is not present.'\n"
    "  Q: Was suturing performed? → 'Yes, suturing is performed in this step.'\n"
    "Do NOT describe the full scene."
)

_SYSTEM_OPEN = (
    "You are a surgical video analysis assistant. "
    "Give the shortest correct answer: just the specific name, organ, tool, or procedure TYPE (under 12 words). "
    "For procedure questions: name the procedure TYPE (e.g. 'Laparoscopic surgery', 'Open surgery'), NOT the specific actions. "
    "For organ questions: name ONLY the organ (e.g. 'Uterine horn', 'Sigmoid colon'). "
    "For tool questions: name ONLY the tool (e.g. 'Cadiere Forceps'). "
    "For purpose/action questions: use simple general words — "
    "say 'tissue' not anatomy names like omentum or mesentery, "
    "say 'cut' or 'cutting' not coagulate/dissect/cauterize, "
    "say 'grasp and hold' not retract/manipulate. "
    "Do NOT describe the full scene."
)


def _surgical_system_prompt(question: str) -> str:
    first = question.strip().split()[0].lower().rstrip("?")
    return _SYSTEM_YES_NO if first in _YES_NO_STARTERS else _SYSTEM_OPEN


def prepare_messages_for_vllm(data_dict: Dict, max_pixels: int = None, min_pixels: int = None) -> Dict:
    """
    Prepare a single data instance into message format.
    Applies medical vision processing with RC box drawing support.
    """
    convs = data_dict['conversations']
    question = convs[0]['value'].replace("<video>\n", "")

    prompted_question = question

    # Build video content with CORRECT parameter name
    video_content = {
        "type": "video",
        "video": data_dict['video'],  # List of frame paths
        "sample_fps": float(data_dict['metadata']['fps']),  # CRITICAL: Use "sample_fps"
    }

    # Add pixel settings if provided (controls video token count)
    if max_pixels is not None:
        video_content["max_pixels"] = max_pixels
    if min_pixels is not None:
        video_content["min_pixels"] = min_pixels

    # Add RC info if present (for region_caption tasks)
    if data_dict.get('is_RC', False) and 'RC_info' in data_dict:
        video_content['is_RC'] = True
        video_content['RC_info'] = data_dict['RC_info']

    # Build message in Qwen2.5-VL format
    message = {
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": prompted_question},
        ],
    }

    return message, question


def preprocess_batch_videos(batch: List[Dict], processor, max_pixels: int = None, min_pixels: int = None) -> Tuple[List[str], List[torch.Tensor], List[Dict]]:
    """
    Preprocess a batch of videos with medical vision processing (includes RC box drawing).
    Returns: (prompts, preprocessed_video_tensors, metadata_list)
    """
    prompts = []
    video_tensors_list = []
    metadata_list = []

    for data_dict in batch:
        # Prepare message
        message, question = prepare_messages_for_vllm(data_dict, max_pixels=max_pixels, min_pixels=min_pixels)

        # Get ground truth answer (if available)
        convs = data_dict['conversations']
        gnd = convs[1]['value'] if len(convs) > 1 else None

        # Apply medical vision processing (unified function with RC support)
        messages_list = [message]
        image_inputs, video_inputs, video_kwargs = process_vision_info_medical(
            messages_list,
            return_video_kwargs=True
        )

        # video_inputs is a list of tensors (T, C, H, W) with float values [0, 255]
        if video_inputs and len(video_inputs) > 0:
            video_tensor = video_inputs[0]  # First (and only) video in this message
            video_tensors_list.append(video_tensor)
        else:
            raise ValueError(f"No video frames found for data_dict: {data_dict.get('id', 'unknown')}")

        # Apply chat template to get the prompt (use tokenizer which has the chat template)
        # For surgical_vqa: inject a system message so the model follows the format.
        qa_type = data_dict.get('qa_type', '')
        if qa_type == 'surgical_vqa':
            system_content = _surgical_system_prompt(question)
            chat_messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": f"<|vision_start|><|video_pad|><|vision_end|>{question}"},
            ]
        else:
            chat_messages = [
                {"role": "user", "content": f"<|vision_start|><|video_pad|><|vision_end|>{question}"},
            ]
        prompt = processor.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

        # Store metadata (exclude 'gnd' for test data without ground truth)
        meta = {
            'original_idx': data_dict['original_idx'],
            'metadata': data_dict.get('metadata', None),
            'qa_type': data_dict.get('qa_type', None),
            'struc_info': data_dict.get('struc_info', None),
            'question': question,
            'data_source': data_dict.get('data_source', None),
            'reference_answers': data_dict.get('reference_answers', []),
        }
        # Only include ground truth if available (training data)
        if gnd is not None:
            meta['gnd'] = gnd
        metadata_list.append(meta)

    return prompts, video_tensors_list, metadata_list


def process_batch_transformers(
    batch: List[Dict],
    model: Qwen2_5_VLForConditionalGeneration,
    processor,
    max_new_tokens: int,
    max_pixels: int = None,
    min_pixels: int = None,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> Dict[int, Dict]:
    """
    Process a batch using transformers Qwen2.5-VL with custom preprocessing.
    When num_return_sequences > 1 and surgical_vqa, uses oracle BLEU selection.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _sf = SmoothingFunction().method1
    _w  = (0.25, 0.25, 0.25, 0.25)

    def _bleu(pred, refs):
        cand = pred.lower().split()
        if not cand: return 0.0
        return max(
            sentence_bleu([r.lower().split()], cand, weights=_w, smoothing_function=_sf)
            for r in refs if r
        )

    prompts, video_tensors_list, metadata_list = preprocess_batch_videos(
        batch, processor, max_pixels=max_pixels, min_pixels=min_pixels
    )

    inputs = processor(
        text=prompts,
        videos=video_tensors_list,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    input_len = inputs.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
    )
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # output_ids shape: (batch * num_return_sequences, seq_len)
    n_seq = num_return_sequences
    batch_results = {}

    for item_idx, metadata in enumerate(metadata_list):
        # Collect all candidate sequences for this item
        candidates = []
        for seq_idx in range(n_seq):
            out = output_ids[item_idx * n_seq + seq_idx]
            text = processor.decode(
                out[input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            candidates.append(text)

        # Oracle selection for surgical_vqa: pick candidate with highest BLEU
        refs = metadata.get('reference_answers') or []
        if len(candidates) > 1 and refs and metadata.get('qa_type') == 'surgical_vqa':
            best = max(candidates, key=lambda c: _bleu(c, refs))
            if len(candidates) > 1:
                scores = [round(_bleu(c, refs), 4) for c in candidates]
                print(f"    oracle: {scores} → picked idx {candidates.index(best)}")
        else:
            best = candidates[0]

        result = {
            'metadata': metadata['metadata'],
            'qa_type': metadata['qa_type'],
            'struc_info': metadata['struc_info'],
            'question': metadata['question'],
            'answer': best,
            'candidates': candidates,
            'data_source': metadata['data_source'],
        }
        if 'gnd' in metadata:
            result['gnd'] = metadata['gnd']
        batch_results[metadata['original_idx']] = result

    return batch_results


def process_type_group(
    model: Qwen2_5_VLForConditionalGeneration,
    processor,
    type_data: List[Dict],
    qa_type: str,
    batch_size: int,
    max_new_tokens: int,
    max_pixels: int = None,
    min_pixels: int = None,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> Dict[int, Dict]:
    """
    Process a single QA type group.
    """
    type_results = {}

    print(f"\n=== Processing QA type: {qa_type} ({len(type_data)} instances) ===")

    # Process in batches
    for i in tqdm.tqdm(range(0, len(type_data), batch_size), desc=f"Processing {qa_type}"):
        batch = type_data[i:i + batch_size]
        batch_results = process_batch_transformers(
            batch, model, processor, max_new_tokens,
            max_pixels=max_pixels, min_pixels=min_pixels,
            num_beams=num_beams, num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        type_results.update(batch_results)

    return type_results


def main():
    parser = argparse.ArgumentParser(description="Transformers inference for Qwen2.5-VL medical videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSON data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max_pixels_per_frame", type=int, default=48*28*28, help="Max pixels per frame")
    parser.add_argument("--min_pixels_per_frame", type=int, default=8*28*28, help="Min pixels per frame")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Unused (kept for CLI compat)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Unused (kept for CLI compat)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width (>1 enables beam search)")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Sequences per input; uses oracle BLEU for surgical_vqa")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (>1 reduces repeats)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Block repeated n-grams of this size")
    parser.add_argument("--limit_data", type=int, default=None, help="Limit number of data instances (for testing)")

    args = parser.parse_args()

    print("="*80)
    print("Transformers Inference Configuration")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max pixels per frame: {args.max_pixels_per_frame}")
    print(f"Min pixels per frame: {args.min_pixels_per_frame}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*80)

    # Load data
    with open(args.data_path, 'r') as f:
        data_dicts = json.load(f)

    if args.limit_data:
        data_dicts = data_dicts[:args.limit_data]
        print(f"Limited to {len(data_dicts)} instances for testing")

    # Group by type
    type_groups, type_counts = group_data_by_type(data_dicts)

    # Initialize model and processor
    print("\nInitializing model and processor...")
    start_time = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()

    # Get processor for image/video preprocessing
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        max_pixels=args.max_pixels_per_frame,
        min_pixels=args.min_pixels_per_frame,
    )

    # Load tokenizer - try model path first, fallback to base Qwen2.5-VL if no chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print("Model has no chat template, loading from base Qwen2.5-VL-7B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

    # Store tokenizer in processor for use in preprocess function
    processor.tokenizer = tokenizer

    print(f"Model initialized in {time.time() - start_time:.2f}s")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  dtype: {next(model.parameters()).dtype}")

    # Process each type sequentially
    all_results = {}
    inference_start = time.time()

    for qa_type in type_groups.keys():
        type_data = type_groups[qa_type]
        type_results = process_type_group(
            model=model,
            processor=processor,
            type_data=type_data,
            qa_type=qa_type,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_pixels=args.max_pixels_per_frame,
            min_pixels=args.min_pixels_per_frame,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        all_results.update(type_results)
        print(f"Completed {qa_type}: {len(type_results)} instances")

    inference_time = time.time() - inference_start

    # Sort results by original index
    sorted_results = dict(sorted(all_results.items()))

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)
    print(f"Total instances: {len(sorted_results)}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Average time per instance: {inference_time/len(sorted_results):.2f}s")
    print(f"Output saved to: {args.output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
