import gc
import os
import sys
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第一张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' # 指定使用第三、四张显卡进行多卡训练
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import torch

from ultralytics import YOLO

if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    # 五个加入 DySampleFusion 和分组归一化 MLLA 的模型，按此顺序串行训练。
    yaml_list = [
        root / "ultralytics/cfg/models/improve/model2025/yolo11/yolo11n-DySampleFusion-MLLAGNorm.yaml",
        root / "ultralytics/cfg/models/improve/model2025/yolo12/yolo12n-DySampleFusion-MLLAGNorm.yaml",
        root / "ultralytics/cfg/models/improve/model2025/yolo26/yolo26n-DySampleFusion-MLLAGNorm.yaml",
        root / "ultralytics/cfg/models/improve/model2025/yolov8/yolov8n-DySampleFusion-MLLAGNorm.yaml",
        root / "ultralytics/cfg/models/improve/model2025/yolov10/yolov10n-DySampleFusion-MLLAGNorm.yaml",
    ]

    # 所有模型使用完全一致的训练参数，保证结果可比较。
    base_args = {
        "data": str(root / "huafen_data6.yaml"),
        "cache": False,
        "imgsz": 640,
        "epochs": 200,
        "batch": 16,
        "close_mosaic": 0,
        "workers": 4,
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "patience": 50,
        "amp": True,
        "cos_lr": False,
        "save_period": -1,
        "project": str(root / "runs/model2025_compare"),
        # trainer=AFSSDetectionTrainer,
        # afss=True,
        # afss_save_refresh_json=False,
        # afss_warmup_epochs=20,
        # afss_update_interval=5,
        # afss_easy_ratio=0.02,
        # afss_moderate_ratio=0.40,
        # afss_easy_forced_gap=10,
        # afss_moderate_forced_gap=3,
        "cls_loss": "slide",
        "iou_loss": "ciou",
        "iou_aux": "none",
        "iou_aux_ratio": 0.5,
    }

    for index, yaml_path in enumerate(yaml_list, start=1):
        if not yaml_path.is_file():
            raise FileNotFoundError(f"模型配置不存在: {yaml_path}")

        exp_name = yaml_path.stem
        optimizer = "MuSGD" if "yolo26" in exp_name else "SGD"
        print(f"\n>>> [{index}/{len(yaml_list)}] 开始训练: {exp_name}", flush=True)

        model = None
        try:
            model = YOLO(str(yaml_path))
            model.train(name=exp_name, optimizer=optimizer, **base_args)
            print(
                f"\n>>> [{index}/{len(yaml_list)}] {exp_name} 训练完成，"
                f"保存路径: {Path(base_args['project']) / exp_name}",
                flush=True,
            )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n>>> 五个模型已全部训练完成。", flush=True)
