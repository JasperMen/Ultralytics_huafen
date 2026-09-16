import warnings, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定使用第一张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' # 指定使用第三、四张显卡进行多卡训练
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer
from ultralytics.models.yolo.segment.afss_train import AFSSSegmentationTrainer
from ultralytics.models.yolo.pose.afss_train import AFSSPoseTrainer
from ultralytics.models.yolo.obb.afss_train import AFSSOBBTrainer

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 全流程实战教程：从零开始完成环境配置、数据集解析、模型训练到测试验证 https://www.bilibili.com/video/BV1tUFkzSEn7/

if __name__ == '__main__':
    # 所有要训练的 yaml 配置文件列表
    yaml_list = [
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/11/yolo11.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-ACA-1.yaml',   #attention
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-ACAB-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-CASAB-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-ContrastDrivenFeatureAggregation-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-CoordAtt-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-DeformableLKA-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-DHPF-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-EMA-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-FSDA-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-KSFA-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-LSKBlock-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-MCA-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-MLCA-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-MultiSEAM-1.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/attention/yolo11/yolo11-SimAM-1.yaml',

        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-ADown.yaml',     #downsample
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-DRFD.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-EdgeLAWDS.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-FreqLAWDS.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-FSCGD.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-FSConv.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-GCNet.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-HWD.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-LAWDS.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-RouterLAWDS.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-SPDConv.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-V7Down.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/downsample/yolo11/yolo11-WaveletPool.yaml',
        
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-CAFM.yaml',   #featurefusion
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-CGAFusion.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-CGFM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-CIDAF.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-CSFCN.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-DAF.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-DCGRM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-DPCF.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-ERM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-FAAFusion.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-GDSAFusion.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-HAFFormer.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-HFFE.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-LCA.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-LowFrequencyFeatureFusion.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-MFM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-MFPM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-MPCA.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-MSAM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-MSCRM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-MSGA.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-PSFM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-PST.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-SDFM.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-SFSFusion.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/improve/featurefusion/yolo11/yolo11-WDAF.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/v8/yolov8.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/11/yolo11.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/12/yolo12.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/v10/yolov10n.yaml',
        # '/home/user/Men/Ultralytics_huafen/ultralytics/cfg/models/26/yolo26.yaml',
        '/home/user/Men/Ultralytics_huafen/改进yaml/yolo11m-CIDAF.yaml',
        '/home/user/Men/Ultralytics_huafen/改进yaml/yolo11m-FSDA-1.yaml',
        '/home/user/Men/Ultralytics_huafen/改进yaml/yolo11m-SFSConv.yaml',
        '/home/user/Men/Ultralytics_huafen/改进yaml/yolo11m-AB.yaml',
        '/home/user/Men/Ultralytics_huafen/改进yaml/yolo11m-ABC.yaml',
        # 继续添加更多 yaml 文件路径...
    ]

    # 依次训练每个 yaml 配置
    for yaml_path in yaml_list:
        # 基础训练参数（所有实验共享）
        base_args = dict(
            data='huafen_data6.yaml', # 数据集配置文件路径
            cache=False, # 是否缓存图像到内存以加快训练速度。False=不缓存，True=缓存到RAM(很吃内存，内存少的慎开)，'disk'=缓存到磁盘(吃硬盘空间)
            imgsz=640, # 输入图像尺寸（像素）
            epochs=300, # 训练总轮数
            batch=8, # 批次大小
            close_mosaic=0, # 最后多少个 epoch 关闭 Mosaic 数据增强。设置 0 代表全程开启 Mosaic 训练
            workers=4, # 数据加载的工作线程数。Windows 下出现卡顿或奇怪错误可尝试设置为 0
            device=os.environ.get("CUDA_VISIBLE_DEVICES", 0), # 训练设备选择，不在这里设置，在头部设置，详细可以看UserGuide.md中的常见问题第4点
            optimizer='MuSGD' if 'yolo26' in yaml_path else 'SGD', # 优化器选择。YOLO26 使用官方推荐的 MuSGD，其他模型使用 SGD
            patience=50, # 早停机制的耐心值。连续 50 个 epoch 验证指标未提升则停止训练。设置 0 关闭早停
            # resume=True, # 断点续训，需要在 YOLO 初始化时加载 last.pt 权重文件
            amp=False, # 是否启用自动混合精度（Automatic Mixed Precision）训练，默认为 True | loss出现nan可以关闭amp
            # fraction=0.2, # 设置0.2代表只选择百分之20的数据进行训练
            cos_lr=False, # 是否使用余弦退火学习率调度器，默认为 False
            save_period=-1, # 每隔多少个 epoch 保存一次 checkpoint（默认 -1 表示禁用，仅保存最好和最后的）
            project='compare', # 训练结果保存的项目目录
            # name 由调用处的 name=exp_name 显式传入（用 yaml 文件名作为实验名）

            # trainer=AFSSDetectionTrainer,
            # afss=True, # 开启 AFSS
            # afss_save_refresh_json=False,
            # afss_warmup_epochs=20, # 前 20 个 epoch 使用全量训练集 warmup
            # afss_update_interval=5, # 每隔 5 个 epoch 刷新一次图像难度状态
            # afss_easy_ratio=0.02, # easy 样本每轮保留 2%
            # afss_moderate_ratio=0.40, # moderate 样本每轮保留 40%
            # afss_easy_forced_gap=10, # easy 样本超过 10 个 epoch 未使用则强制回看
            # afss_moderate_forced_gap=3, # moderate 样本超过 3 个 epoch 未使用则强制覆盖
            # afss_thresholds={
            #     "detect": [0.55, 0.85],
            #     "obb": [0.55, 0.85],
            #     "segment": [0.55, 0.85],
            #     "pose": [0.55, 0.85],
            # },

            # -------------------- LOSS部分(更多解释可以看LOSS-UserGuide.md) --------------------
            cls_loss='bce', # 分类损失类型可选：bce, slide, ema_slide, focal, varifocal, qualityfocal
            iou_loss='ciou', # IoU损失可选：基础 iou/giou/diou/ciou/eiou/siou/shapeiou/piou/piou2；组合 inner_<base>/focaler_<base>；MPD mpdiou/inner_mpdiou/focaler_mpdiou；Wise wiseiou[_inner|_focaler]_<variant>
            iou_aux='none', # IoU辅助分支可选：none, gcd, nwd（none 表示关闭辅助分支）
            iou_aux_ratio=0.5, # IoU主损失与辅助分支混合系数（0~1），仅在 iou_aux != none 时生效
        )

        model = YOLO(yaml_path)
        exp_name = yaml_path.split('/')[-1].replace('.yaml', '')
        save_dir = os.path.join('train', exp_name)
        print(f"\n>>> 开始训练: {exp_name}")
        model.train(name=exp_name, **base_args)
        print(f"\n>>> {exp_name} 训练完成，保存路径: {save_dir}")
