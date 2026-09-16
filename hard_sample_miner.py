#!/usr/bin/env python3
"""
YOLO 难样本挖掘脚本
用法: python hard_sample_miner.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import yaml
from ultralytics import YOLO

# ──────────────── 配置 ────────────────
WEIGHT_PATH = "/home/user/Men/Ultralytics_huafen/runs/detect/train/exp4/weights/best.pt"
DATA_YAML   = "/home/user/Men/Ultralytics_huafen/data.yaml"
OUTPUT_DIR  = Path("/home/user/Men/Ultralytics_huafen/runs/detect/hard_samples_exp4")
TOP_N       = 50


def calculate_iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1, a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def load_yolo_label(label_path, img_w=640, img_h=640):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1, y1 = cx * img_w - w * img_w / 2, cy * img_h - h * img_h / 2
            x2, y2 = cx * img_w + w * img_w / 2, cy * img_h + h * img_h / 2
            boxes.append({"box": [x1, y1, x2, y2], "cls": cls})
    return boxes


def img_to_label(img_path):
    """images/val/xxx.jpg -> labels/val/xxx.txt"""
    p = img_path.replace("/images/", "/labels/")
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        p = p.replace(ext, ".txt")
    return p


def match_boxes(gt, pred, iou_thr=0.5):
    matched, fp, fn = set(), [], []
    ious = []
    for pi, pp in enumerate(sorted(pred, key=lambda x: x["conf"], reverse=True)):
        best_i, best_io = -1, 0.0
        for gi, gg in enumerate(gt):
            if gi in matched:
                continue
            io = calculate_iou(pp["box"], gg["box"])
            if io > best_io:
                best_io, best_i = io, gi
        if best_io >= iou_thr:
            matched.add(best_i)
            ious.append(best_io)
        else:
            ious.append(0.0)
            fp.append(pi)
    fn = [gi for gi in range(len(gt)) if gi not in matched]
    return list(matched), fp, fn, ious


def analyze_image(model, img_path, class_names):
    results = model.predict(source=img_path, conf=0.001, iou=0.7, verbose=False, device=0)
    if not results:
        return None
    r = results[0]
    pred = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            pred.append({"box": [x1, y1, x2, y2], "conf": float(box.conf[0]), "cls": int(box.cls[0])})
    gt = load_yolo_label(img_to_label(img_path))
    matched, fp, fn, ious = match_boxes(gt, pred)
    tp = len(matched)
    recall = tp / len(gt) if gt else (1.0 if not pred else 0.0)
    precision = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    miss_rate = len(fn) / len(gt) if gt else 0.0
    difficulty = miss_rate * 0.5 + (1 - precision) * 0.3 + max(0, 1 - avg_iou) * 0.2
    return {
        "img_path": img_path,
        "img_name": os.path.basename(img_path),
        "num_gt": len(gt), "num_tp": tp, "num_fn": len(fn), "num_fp": len(fp),
        "recall": recall, "precision": precision,
        "avg_iou": avg_iou, "difficulty": difficulty,
    }


def get_val_images(data_yaml):
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    val_dir = root / cfg.get("val", "images/val")
    if not val_dir.exists():
        val_dir = root
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
        images.extend(val_dir.glob(f"**/{ext}"))
    return [str(p) for p in sorted(images)], cfg.get("names", {})


def main():
    print("=" * 60)
    print("YOLO 难样本挖掘")
    print("=" * 60)

    print(f"\n[1/4] 加载模型...")
    model = YOLO(WEIGHT_PATH)

    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", {})
    print(f"    类别: {len(class_names)} 类")

    print(f"\n[2/4] 扫描验证集...")
    val_images, _ = get_val_images(DATA_YAML)
    print(f"    共 {len(val_images)} 张图片")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/4] 正在分析每张图片...")
    scores = []
    for i, img in enumerate(val_images):
        if i % 20 == 0:
            print(f"    进度: {i}/{len(val_images)} ({100*i/len(val_images):.1f}%)")
        r = analyze_image(model, img, class_names)
        if r:
            scores.append(r)
    print(f"    完成! 共分析 {len(scores)} 张")

    by_diff  = sorted(scores, key=lambda x: x["difficulty"], reverse=True)
    by_fn    = sorted(scores, key=lambda x: x["num_fn"], reverse=True)
    by_fp    = sorted(scores, key=lambda x: x["num_fp"], reverse=True)
    by_recall = sorted(scores, key=lambda x: x["recall"])

    # ── 报告 ──
    report = []
    report.append("# YOLO 难样本分析报告\n")
    report.append(f"**模型**: `{WEIGHT_PATH}`  ")
    report.append(f"**验证集图片数**: {len(val_images)}  \n")

    report.append("## 汇总\n")
    report.append(f"| 指标 | 数值 |")
    report.append(f"|------|------|")
    report.append(f"| 总 GT 框 | {sum(s['num_gt'] for s in scores)} |")
    report.append(f"| 总漏检 FN | {sum(s['num_fn'] for s in scores)} |")
    report.append(f"| 总误检 FP | {sum(s['num_fp'] for s in scores)} |")
    report.append(f"| 平均难度分 | {sum(s['difficulty'] for s in scores)/len(scores):.4f} |")
    report.append("")

    report.append(f"## 最难样本 Top {TOP_N}（综合难度分数）\n")
    report.append("| 排名 | 图片名 | 难度分 | 召回率 | 精确率 | 漏检 | 误检 |")
    report.append("|------|--------|--------|--------|--------|------|------|")
    for rank, s in enumerate(by_diff[:TOP_N], 1):
        report.append(f"| {rank} | `{s['img_name']}` | {s['difficulty']:.4f} | "
                      f"{s['recall']:.3f} | {s['precision']:.3f} | {s['num_fn']} | {s['num_fp']} |")

    report.append(f"\n## 漏检最多 Top {TOP_N}\n")
    report.append("| 排名 | 图片名 | 漏检数 | 召回率 | GT数 |")
    report.append("|------|--------|--------|--------|------|")
    for rank, s in enumerate(by_fn[:TOP_N], 1):
        if s['num_fn'] > 0:
            report.append(f"| {rank} | `{s['img_name']}` | **{s['num_fn']}** | "
                          f"{s['recall']:.3f} | {s['num_gt']} |")

    report.append(f"\n## 误检最多 Top {TOP_N}\n")
    report.append("| 排名 | 图片名 | 误检数 | 精确率 | 预测数 |")
    report.append("|------|--------|--------|--------|--------|")
    for rank, s in enumerate(by_fp[:TOP_N], 1):
        if s['num_fp'] > 0:
            report.append(f"| {rank} | `{s['img_name']}` | **{s['num_fp']}** | "
                          f"{s['precision']:.3f} | {s['num_tp'] + s['num_fp']} |")

    report.append("\n## 召回率最低 Top 20\n")
    report.append("| 排名 | 图片名 | 召回率 | 漏检 | GT数 |")
    report.append("|------|--------|--------|------|------|")
    for rank, s in enumerate(by_recall[:20], 1):
        if s['num_fn'] > 0:
            report.append(f"| {rank} | `{s['img_name']}` | **{s['recall']:.3f}** | "
                          f"{s['num_fn']} | {s['num_gt']} |")

    report.append("\n## 重点检查建议\n")
    report.append("以下图片建议优先人工核查标注质量：\n")
    for s in by_diff[:20]:
        reasons = []
        if s['num_fn'] >= 3: reasons.append(f"严重漏检(FN={s['num_fn']})")
        if s['num_fp'] >= 3: reasons.append(f"严重误检(FP={s['num_fp']})")
        if s['recall'] < 0.3: reasons.append(f"召回极低({s['recall']:.2f})")
        if reasons:
            report.append(f"- `{s['img_name']}`: {', '.join(reasons)}\n")

    # 保存
    report_path = OUTPUT_DIR / "hard_samples_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    with open(OUTPUT_DIR / "hard_samples_data.json", "w", encoding="utf-8") as f:
        json.dump({"by_difficulty": by_diff, "by_fn": by_fn, "by_fp": by_fp}, f,
                   ensure_ascii=False, indent=2, default=str)

    with open(OUTPUT_DIR / "hard_sample_paths.txt", "w") as f:
        for s in by_diff[:100]:
            f.write(f"{s['img_path']}\n")

    print(f"\n[4/4] 完成！")
    print(f"  📄 报告: {report_path}")
    print(f"  📊 JSON: {OUTPUT_DIR / 'hard_samples_data.json'}")
    print(f"  📋 路径清单: {OUTPUT_DIR / 'hard_sample_paths.txt'}")

    print("\n最难样本 Top 20 预览:")
    print(f"{'排名':<4} {'图片名':<42} {'难度分':>8} {'漏检':>4} {'误检':>4} {'召回':>6}")
    print("-" * 72)
    for rank, s in enumerate(by_diff[:20], 1):
        print(f"{rank:<4} {s['img_name']:<42} {s['difficulty']:>8.4f} "
              f"{s['num_fn']:>4} {s['num_fp']:>4} {s['recall']:>6.3f}")


if __name__ == "__main__":
    main()
