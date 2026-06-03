"""
Convert SURGVU25 Cat.2 dataset to MedVidBench frame format.

Input layout:
    SURGVU25_cat_2_sample_set_public/
        caseXXX/
            caseXXX.mp4
            caseXXX_question.json   # single question string
            caseXXX.json            # list of 5 reference answers

Output:
    data/testdata/SURGVU25/caseXXX/frame_000000.jpg  ...
    data/test_data_surgvu25.json  (MedVidBench list format)

Each record in the output JSON:
    {
      "id": "case122",
      "qa_type": "surgical_vqa",
      "metadata": {"video_id": "case122", "fps": "1.0", ...},
      "conversations": [
          {"from": "human", "value": "<video>\n<question>"},
          {"from": "gpt",   "value": "<first reference answer>"}
      ],
      "reference_answers": ["ans1", "ans2", "ans3", "ans4", "ans5"],
      "video": ["data/testdata/SURGVU25/case122/frame_000000.jpg", ...],
      "struc_info": null,
      "data_source": "SURGVU25"
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.io as tvio
from PIL import Image


EXTRACT_FPS = 1.0  # 1 frame per second — matches MedVidBench default


def extract_frames(mp4_path: Path, out_dir: Path, target_fps: float = EXTRACT_FPS) -> tuple[list[str], float, int]:
    """Extract frames from MP4 at target_fps, save as JPG. Returns (frame_paths, source_fps, total_frames)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    video, _, info = tvio.read_video(str(mp4_path), pts_unit="sec", output_format="TCHW")
    source_fps = float(info["video_fps"])
    total_frames = video.shape[0]

    step = max(1, round(source_fps / target_fps))
    indices = list(range(0, total_frames, step))

    frame_paths = []
    for seq_idx, frame_idx in enumerate(indices):
        frame_tensor = video[frame_idx]  # (C, H, W) uint8
        frame_np = frame_tensor.permute(1, 2, 0).numpy()
        img = Image.fromarray(frame_np)
        fname = out_dir / f"frame_{seq_idx:06d}.jpg"
        img.save(str(fname), quality=95)
        frame_paths.append(str(fname))

    return frame_paths, source_fps, total_frames


def process_case(case_dir: Path, frames_root: Path, target_fps: float = EXTRACT_FPS) -> dict:
    case_id = case_dir.name

    mp4 = case_dir / f"{case_id}.mp4"
    q_file = case_dir / f"{case_id}_question.json"
    a_file = case_dir / f"{case_id}.json"

    if not mp4.exists():
        raise FileNotFoundError(f"Missing video: {mp4}")
    if not q_file.exists():
        raise FileNotFoundError(f"Missing question: {q_file}")
    if not a_file.exists():
        raise FileNotFoundError(f"Missing answers: {a_file}")

    question = json.loads(q_file.read_text().strip())
    if not isinstance(question, str):
        question = str(question)

    answers = json.loads(a_file.read_text().strip())
    if not isinstance(answers, list):
        answers = [str(answers)]

    out_dir = frames_root / case_id
    print(f"  [{case_id}] extracting frames from {mp4.name} → {out_dir}")
    frame_paths, source_fps, total_frames = extract_frames(mp4, out_dir, target_fps)
    print(f"    source fps={source_fps:.2f}, total={total_frames} frames → extracted {len(frame_paths)} at {EXTRACT_FPS} fps")

    return {
        "id": case_id,
        "qa_type": "surgical_vqa",
        "metadata": {
            "video_id": case_id,
            "fps": str(EXTRACT_FPS),
            "source_fps": str(round(source_fps, 4)),
            "input_video_start_frame": "0",
            "input_video_end_frame": str(len(frame_paths) - 1),
        },
        "conversations": [
            {"from": "human", "value": f"<video>\n{question}"},
            {"from": "gpt",   "value": answers[0]},
        ],
        "reference_answers": answers,
        "video": frame_paths,
        "struc_info": None,
        "data_source": "SURGVU25",
    }


def main():
    ap = argparse.ArgumentParser(description="Convert SURGVU25 Cat.2 → MedVidBench frame format")
    ap.add_argument("--input_dir", default="SURGVU25_cat_2_sample_set_public",
                    help="Path to SURGVU25 dataset root")
    ap.add_argument("--frames_dir", default="data/testdata/SURGVU25",
                    help="Where to save extracted frames")
    ap.add_argument("--output_json", default="data/test_data_surgvu25.json",
                    help="Output JSON path")
    ap.add_argument("--fps", type=float, default=EXTRACT_FPS,
                    help="Target extraction fps (default: 1.0)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    frames_root = Path(args.frames_dir)
    output_json = Path(args.output_json)

    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    case_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("case")])
    if not case_dirs:
        print(f"ERROR: no case* subdirectories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(case_dirs)} cases: {[d.name for d in case_dirs]}")
    print(f"Frames → {frames_root}")
    print(f"Output → {output_json}")
    print()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    target_fps = args.fps

    records = []
    for case_dir in case_dirs:
        try:
            record = process_case(case_dir, frames_root, target_fps)
            records.append(record)
        except Exception as e:
            print(f"  ERROR processing {case_dir.name}: {e}", file=sys.stderr)

    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nDone. {len(records)} records written to {output_json}")

    total_frames = sum(len(r["video"]) for r in records)
    print(f"Total frames extracted: {total_frames}")


if __name__ == "__main__":
    main()
