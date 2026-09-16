#!/usr/bin/env python3
"""
将 YOLO_Dataset_Split5 标签中的不连续 ID 映射为连续 ID：
  原 ID 17(syh 芍药花) → 新 ID 16
  原 ID 18(yb 圆柏)   → 新 ID 17
  原 ID 19(ych 油菜花) → 新 ID 18
其余 0~15 保持不变。
同时生成修正后的 huafen_data.yaml。.
"""

import os

BASE = "/home/user/Men/Huafen_Dataset/YOLO_Dataset_Split5"
YAML_PATH = os.path.join(BASE, "huafen_data.yaml")

# ID 重映射表（旧ID → 新ID）
REMAP = {
    17: 16,  # syh 芍药花
    18: 17,  # yb 圆柏
    19: 18,  # ych 油菜花
}


def remap_label_file(path):
    with open(path) as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            old_cls = int(parts[0])
            if old_cls in REMAP:
                parts[0] = str(REMAP[old_cls])
                changed = True
        new_lines.append(" ".join(parts) + "\n" if parts else line)

    if changed:
        with open(path, "w") as f:
            f.writelines(new_lines)
    return changed


def main():
    label_base = os.path.join(BASE, "labels")
    total_files = 0
    changed_files = 0

    for split in ["train", "val", "test"]:
        d = os.path.join(label_base, split)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".txt"):
                continue
            total_files += 1
            path = os.path.join(d, fname)
            if remap_label_file(path):
                changed_files += 1

    print(f"处理完成: 共 {total_files} 个标签文件，修改了 {changed_files} 个")

    # 写入修正后的 yaml
    new_yaml = f"""# 花粉检测数据集配置

# 数据集根目录
path: {BASE}

# 训练、验证、测试图片路径（相对于path）
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 19

# 类别名称（根据 YOLO_Dataset_Split5 实际标签 + 数据集.md 映射修正）
# 标签 ID 已统一为 0~18 连续编号
names:
  0: azs      # 矮紫杉
  1: bjys     # 北京云杉
  2: cmx      # 草木犀
  3: dzh      # 大籽蒿
  4: gsh      # 格桑花
  5: gybc     # 狗尾巴草
  6: hcm      # 黄刺玫
  7: jkpgy    # 菊科蒲公英
  8: jkzhkmc  # 菊科中华苦荬菜
  9: jyh      # 金银花
  10: lk      # 藜科
  11: ls      # 柳树
  12: not_pollen  # 杂质
  13: qwkpg   # 蔷薇科苹果
  14: qxlm    # 球悬铃木
  15: sb      # 松柏
  16: syh     # 芍药花
  17: yb      # 圆柏
  18: ych     # 油菜花
"""
    with open(YAML_PATH, "w") as f:
        f.write(new_yaml)
    print(f"已写入修正后的 yaml: {YAML_PATH}")


if __name__ == "__main__":
    main()
