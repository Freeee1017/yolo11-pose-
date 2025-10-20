# YOLO11 婴儿关键点检测（21 关键点）

本项目实现了基于 YOLO 思想的婴儿单目标关键点检测（21 点），支持训练、验证、单图预测与可视化。项目已做了训练器/验证器/预测器的模块化封装，开箱即用。

- 任务：单类别 baby + bbox + 21×(x,y,vis)
- 标签格式：YOLO 风格，1 行 68 个字段 = 1 类别 + 4 框(cx,cy,w,h) + 63 关键点(每点 x,y,vis)
- 数据组织：dataset/{images,labels}/{train,val,test}


## 功能特点
- 训练器 YOLOTrainer：Cosine LR、可选 Warmup、早停、多指标记录、CSV/JSON 导出
- 验证器 YOLOValidater：按原图坐标计算 IoU、关键点 MAE，支持可视化与 per-image CSV
- 预测器 YOLOPredicter：单图/批量预测，关键点/框自动从 letterbox 还原回原图
- 可视化 YOLOVisualizer：绘制 bbox 与 21 个关键点（含名称/索引可选）
- 配置集中化：`config.yaml` 一处管理数据路径、训练策略、增强、损失权重、推理策略等


## 项目结构
- `my_train.py`：启动训练（读取 `config.yaml`）
- `my_test.py`：在测试集上评估并导出 per-image CSV
- `my_predict.py`：单图预测与可视化保存
- `trainer.py`：训练器实现（保存权重、日志、CSV/JSON 指标等）
- `validater.py`：验证器实现（IoU、关键点 MAE、可视化、CSV 导出）
- `predicter.py`：推理与坐标映射封装
- `visualize.py`：可视化工具
- `yolo_dataset.py`：数据集、DataLoader 与增强器接入
- `model.py`、`backbone.py`、`neck.py`、`head.py`、`loss.py`：模型与损失
- `config.yaml`：训练/推理的集中配置
- `runs/`：训练产出（日志、权重、验证可视化、预测结果等）


## 环境要求
- Python 3.8+
- PyTorch（请按官方指引安装以匹配本机 CUDA/CPU）
- 其它依赖见 `requirements.txt`

安装步骤（Windows cmd）：

```cmd
:: 1) 先安装与您环境匹配的 PyTorch（见 https://pytorch.org/ ）

:: 2) 其余依赖
pip install -r requirements.txt
```

可选：使用 TensorBoard

```cmd
pip install tensorboard
:: 训练期间将日志写入 runs\logs\<exp_name>
tensorboard --logdir runs\logs
```


## 数据准备
目录结构示例：

```
dataset/
  images/
    train/  *.jpg|*.png
    val/
    test/
  labels/
    train/  *.txt
    val/
    test/
```

每张图片对应一个标签文件，内容 1 行 68 列：

```
<class_id> <cx> <cy> <w> <h>  kpt0_x kpt0_y kpt0_v  kpt1_x kpt1_y kpt1_v ... kpt20_x kpt20_y kpt20_v
```

说明：
- 坐标均为相对原图的归一化数值（0..1）；(cx,cy) 为中心点，(w,h) 为宽高
- 关键点为 21×(x,y,vis)，vis 可用 0/1/2，内部会将 2 视作 1（可见）

在 `config.yaml` 中配置数据路径（示例已给出，可按需修改）：

```yaml
train_img_dir: "f:/dataset/images/train"
train_label_dir: "f:/dataset/labels/train"
val_img_dir: "f:/dataset/images/val"
val_label_dir: "f:/dataset/labels/val"
```


## 快速开始

### 训练
最简单：使用默认配置并自动命名实验目录。

```cmd
python my_train.py --config config.yaml --device auto
```

常用参数：
- `--resume <ckpt>`：从检查点继续训练（支持完整 checkpoint 或仅权重）
- `--device {auto|cpu|cuda:0}`：设备选择
- `--exp-name <name>`：自定义实验名；默认会根据 e/b/lr 与时间戳生成
- `--debug`：调试模式（少量 epoch，快速跑通）
- `--early-stop` / `--no-early-stop`：覆盖配置文件里的早停开关

训练产出默认位于：
- `runs/train/<exp_name>/weights/{last.pt,best.pt,last_weights.pt,best_weights.pt}`
- `runs/logs/<exp_name>/`（TensorBoard 日志）
- `runs/train/<exp_name>/val_vis/`（验证阶段可视化，若启用）
- `runs/train/<exp_name>/results.csv`（按 epoch 追加训练/验证指标）
- `runs/train/<exp_name>/epoch_losses.json`（便于绘图的 epoch 损失记录）

### 验证（内置于训练流程）
训练中每 `val_interval` 个 epoch 验证一次，同时进行早停监控：
- 监控项由 `early_stop_metric` 指定（如 `mean_iou` 或 `val_loss`）
- 方向由 `early_stop_mode` 指定（`min` 或 `max`）
- 改善阈值由 `early_stop_delta` 指定

### 在测试集上评估并导出 CSV
```cmd
python my_test.py ^
  --weights runs\train\<your_exp>\weights\best.pt ^
  --device cuda:0 ^
  --img-size 640 ^
  --batch-size 16 ^
  --num-workers 4 ^
  --test-img-dir dataset\images\test ^
  --test-label-dir dataset\labels\test ^
  --out-csv test_results.csv
```
输出 `test_results.csv` 包含每张图片的得分、IoU、关键点误差等信息。

### 单图预测
```cmd
python my_predict.py ^
  --weights runs\train\<your_exp>\weights\best.pt ^
  --img path\to\image.jpg ^
  --img-size 640 ^
  --device cpu ^
  --save-dir runs\predict ^
  --save-original
```
结果图保存为 `runs/predict/<name>_pred.jpg`（原图可选 `_orig.jpg`）。


## 配置说明（config.yaml）
核心字段（节选）：
- 模型/任务
  - `nc`: 类别数（默认 1）
  - `nk`: 关键点数量（默认 21）
  - `classes`: 类别名称（默认 ["baby"]) 
- 训练
  - `img_size`, `batch_size`, `epochs`, `lr`, `weight_decay`, `num_workers`, `seed`
  - `warmup_epochs`, `cos_lr`
  - 早停：`early_stop`, `patience`, `early_stop_metric`, `early_stop_mode`, `early_stop_delta`
  - 验证：`val_interval`
  - 记录与导出：默认写 CSV/JSON；TensorBoard 若安装则自动启用
- 数据增强
  - `use_augmentation`: 是否启用训练集增强
  - `horizontal_flip_prob`, `rotation_prob`, `scale_prob`, `translation_prob` 等
- 可视化（验证阶段）
  - `val_visualize`: 是否保存验证可视化
  - `val_visual_interval`: 可视化的 epoch 间隔
  - `val_visual_max_images`: 每批保存张数上限
- 损失与关键点
  - `lambda_box`, `lambda_cls`, `lambda_kpt`, `lambda_kpt_vis`, `lambda_dfl`
  - `kpt_vis_threshold`: 可见性阈值（预测/可视化时用于屏蔽低置信关键点）
- 推理策略
  - `topk_fuse`, `topk`, `fuse_temp`：可选的 Top-K 融合（缓解网格量化误差）

提示：`my_train.py` 的 `--early-stop/--no-early-stop/--debug` 可覆盖部分配置。


## 常见问题与排查
1) 数据路径/标签找不到
- 请检查 `config.yaml` 的 train/val 路径以及测试脚本的 `--test-img-dir/--test-label-dir`
- 标签需与图片同名（.txt 对应 .jpg/.png），且为单行 68 列

2) 显存不足 / 训练很慢
- 减小 `batch_size` 或 `img_size`；或切换到 `--device cpu` 以排查逻辑

3) 安装 PyTorch 失败
- 请根据 https://pytorch.org/ 选择与本机 CUDA/CPU 对应的安装命令，再执行 `pip install -r requirements.txt`

4) 预测输出为空或关键点全 0
- 检查 `kpt_vis_threshold` 是否过高；或权重是否正确加载（命令行打印“Loaded weights”）

5) TensorBoard 无日志
- 确认安装 `tensorboard`，并指向 `runs\logs\<exp_name>`


## 开发者提示
- 训练权重将保存到 `runs/train/<exp>/weights/`，包含完整 checkpoint 与仅权重文件，预测/评估均支持加载
- `metrics.csv / epoch_losses.json` 便于二次分析或绘图
- 若希望在训练时开启混合精度，可在配置里添加 `amp: true`（需 GPU）


## 许可证
未显式声明许可证。如需开源分发，请先添加相应 LICENSE。

---
如需我帮你根据你的数据集路径或显卡情况生成一套最小可用命令，告诉我你的显卡与数据所在位置即可。
