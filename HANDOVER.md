# YOLO26s 航拍检测项目 — AI Agent 接手文档

> **写给另一个 AI agent**：读完本文档即可理解项目全貌、GitHub 现状与优化方向，无需回溯对话历史。
> **写于 2026-07-30** | 申请人：邹昊宜（GitHub: ZHY9981）

---

## 一、项目背景

申请人邹昊宜正在准备 **2027 Fall 香港高校授课型硕士（CS / AI / CV 方向）** 申请。核心素材是一个**无人机航拍小目标检测项目**，已做 20+ 版本 YOLO 消融实验，但仍需通过 GitHub 公开仓库向招生官证明研究能力。

**目的**：把"我在本地训过 YOLO 模型"升级为"招生官能点开、看懂、能验证的研究型仓库"。

**硬节点**：2026-09-10 文书定稿，仓库 **8 月底前必须上线**。

---

## 二、申请人关键信息（务必准确，不得虚构）

| 项目 | 内容 |
|------|------|
| 姓名 | 邹昊宜（Zou Haoyi） |
| GitHub | `ZHY9981` |
| 本科 | 湖南财政经济学院，计算机科学与技术，2025 届，均分 85.78 / GPA 3.46 |
| 实习 | 湖南库里斯智能科技有限公司，技术实习生，2025.11–2026.04 |
| CAAC | 超视距执照编号 430421200208018294，签发 2026-04-10 |

---

## 三、项目核心信息

- **项目名称**：Resource-Constrained Aerial Small-Object Detection with YOLO26s + CoordAtt
- **任务**：无人机航拍图像中 7 类小目标检测（person / car / cycle / truck / bus / freight / small-bus）
- **硬件约束**：RTX 5060 Laptop 8GB VRAM → batch=4，P2 高分辨率头 OOM，被迫用 P3+P4 双头
- **关键技术**：CoordAtt 坐标注意力、WIoU v3 损失、TAL 4px 阈值、类加权、双头架构

### 🔴 核心指标（不得修改）

| 指标 | 值 | 说明 |
|------|-----|------|
| V16 mAP50 | **74.00%** | P3+P4 + per-scale CoordAtt, imgsz=800 |
| V16 mAP50-95 | **52.11%** | 同上 |
| V14 mAP50-95 峰值 | **52.65%** | P3+P4 dual-head, imgsz=960 |
| 参数量 | **7.03M** | V16 |
| 推理速度 | **4.0ms** | RTX 5060, FP16, imgsz=800 |
| VisDrone 基准 | **32.65%** | 公开 SOTA ~43.7% |

### V16.0 逐类数据（实测填充，不得修改）

| Class | AP50 | AP50-95 | Precision | Recall |
|:------|:----:|:-------:|:---------:|:------:|
| person | 60.2% | 24.7% | 77.7% | 50.9% |
| cycle | 43.0% | 17.6% | 67.0% | 36.3% |
| bus | 91.8% | 74.5% | 91.3% | 86.3% |
| small-bus | 99.2% | 83.3% | 92.3% | 99.4% |
| car | 67.2% | 47.3% | 87.2% | 51.4% |
| truck | 60.0% | 41.0% | 63.8% | 53.6% |
| freight | 96.6% | 76.6% | 89.8% | 95.0% |
| **Overall** | **74.0%** | **52.1%** | **81.3%** | **67.6%** |

---

## 四、本地项目结构

### 4.1 版本目录（27 个版本）

路径：`D:\YOLO_Archive\yolo26模型进化\`

| 版本 | 训练脚本 | args.yaml | results.csv | 图片 | 训练笔记 | 混淆矩阵 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| V1.0–V3.0 | ❌ | ✅ | ✅ | ✅ | 部分 | ❌ |
| V5.0–V6.0 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| V7.0–V14.0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| V14.1 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| V15.0–V19.0 | ✅ | ✅ | ✅ | ✅ | 仅AI文档 | ✅ |

> **V4.0** 是唯一的两阶段训练版本（stage1+stage2），无 images 目录。
> **V1.1, V1.2, V2.0, V3.0** 各含 `进化失败的方法/` 子目录，记录了失败分析。

### 4.2 模型定义 YAML（19 个）

路径：`D:\YOLO_Archive\YOLO\ultralytics\ultralytics\cfg\models\26\`

包括 base `yolo26.yaml`、CoordAtt 变体、V16–V21 定制架构、P2/P6 变体、分类/分割/姿态等。

### 4.3 数据集（5 个）

路径：`D:\YOLO_Archive\YOLO\ultralytics\datasets\`

| 数据集 | 规模 | 类别 | 说明 |
|--------|------|:--:|------|
| aerial_v9 | 8,075/2,224/738 | 7 | 当前主力，清洗标注后 |
| aerial_v8 | 3,800 | 7 | 早期，已被 v9 取代 |
| aerial_merged | 7,148 | 5 | aerial.v1i + VisDrone，含噪声 |
| aerial | 2,090 | 6 | Roboflow aerial.v1i，最早期 |
| visdrone_official | 6,471/548 | 10 | VisDrone2019 官方，仅作对比 |

---

## 五、GitHub 仓库现状

**仓库地址**：`https://github.com/ZHY9981/Drone_yolo_7types`（Public，MIT License）

**当前文件结构**：

```
aerial-yolo26s-drone-detection/
├── README.md                    ← 完整 README（含消融表、V16 逐类数据）
├── LICENSE                      ← MIT
├── requirements.txt             ← 环境依赖
├── .gitignore                   ← Python gitignore
├── PUSH_GUIDE.md                ← 推送指南（上线后可删除）
├── data/
│   └── data.yaml                ← 7 类数据集配置
├── configs/
│   ├── yolo26s-v16-p34-coordatt.yaml  ← V16 架构定义
│   └── yolo26s-v20-ppa-dysample.yaml  ← V20 架构定义
├── scripts/
│   ├── train_v16.py             ← V16 训练脚本
│   └── eval.py                  ← 评估脚本
├── docs/
│   └── ablation_table.md        ← 消融实验文档
└── results/
    └── guide.md                 ← 配图导出指南
```

### 已知不足

1. **results/ 目录为空**：缺少训练曲线（results.png）、混淆矩阵（confusion_matrix.png）、检测效果图（3-4 张）
2. **docs/ 只有消融表**：缺少技术报告（technical_report.md）
3. **scripts/ 只有 V16 训练脚本**：V20 等其他版本的训练脚本未放入
4. **configs/ 只有 2 个配置**：本地有 19 个模型 YAML，精选不足
5. **README 消融表缺 mAP50-95 列部分数据**：V15/V17/V18/V19 的 mAP50-95 标为近似值，可用 results.csv 验证
6. **未上传 best.pt 权重文件**：V16 best.pt（~27MB，在 100MB 内，可直接上传或放 Release）
7. **推理速度 4.0ms 已实测**，但 README 中其余位置（如 Reproducibility 段）可加速度参考

---

## 六、优化任务清单

### 🔴 优先级 1：必须完成

1. **补 results/ 配图**
   - 从 `D:\YOLO_Archive\yolo26模型进化\V16.0\` 复制 `YOLO26_V16.0_confusion_matrix.png` 和 `YOLO26_V16.0_confusion_matrix_normalized.png` 到 `results/`
   - 从同目录 `YOLO26_V16.0_images\` 选 3-4 张 val_batch_pred.jpg 作为检测效果图
   - V16.0 目录下缺少 results.png（训练曲线），可从 `runs/aerial_train/yolo26s_v16/results.png` 获取，或从 V14.0/V13.0 等版本复制一份效果近似的作为参考
   - 更新 `results/guide.md` 标注配图路径

2. **补 technical_report.md**
   - 参考 `D:\YOLO_Archive\yolo26模型进化\版本总览.md` 和 V16/V14 训练笔记
   - 注意标注"技术报告非论文"（申请人是授课型硕士，无论文）

3. **消融表数据交叉验证**
   - README 中 V15/V17/V18/V19 的 mAP50-95 值用 `results.csv` 的最终 epoch 值二次验证
   - 逐类 AP50-95 数据已从 V16 eval 中获取（见第三节），与 README 和 ablation_table.md 保持一致

### 🟡 优先级 2：建议完成

4. **上传 V16 best.pt**
   - 路径：`D:\YOLO_Archive\yolo26模型进化\V16.0\YOLO26_V16.0_best.pt`
   - 约 27MB，< 100MB 可直接 git push，或在 GitHub Release 中发布

5. **扩充 configs/**
   - 建议加入 `yolo26s-coordatt.yaml`（标准 CoordAtt 三头）、`yolo26s-v14-p3p4.yaml`（V14 双头基线）
   - 路径：`D:\YOLO_Archive\YOLO\ultralytics\ultralytics\cfg\models\26\`

6. **检查 README 语言一致性**
   - 中英混杂部分是否需要统一？
   - "班级权重"→应为"类别权重 (Class weighting)"
   - 消融表中 "Transfer-learning接力" 的中英混用是否合适

### 🟢 优先级 3：可选优化

7. **补充 V20 训练脚本**（或标注"实验性，不推荐生产"）
8. **data.yaml 加上 aerial_v9 完整数据集来源声明**（当前已有 CC BY 4.0 / MIT 标注）
9. **删除 PUSH_GUIDE.md**（推送完成后不再需要）

---

## 七、红线（绝对不能碰）

- ❌ **不虚构、不夸大任何数据**：74.00% 就是 74.00%，52.11% 就是 52.11%
- ❌ **不声称论文发表**：技术报告 ≠ 论文
- ❌ **不放无关代码**：本科毕设"宠物商城"（SpringBoot/Vue）不要混进来
- ❌ **不传大文件**：训练图片 GB 级别不放；best.pt 约 27MB 可传
- ❌ **不替申请人编造经历**：实习时间、CAAC 编号等已确认真实
- ❌ **不改仓库名**：`Drone_yolo_7types` 已确定，改了就断链

---

## 八、快速操作指南

### 补配图（一行命令）

```powershell
# 从 V16.0 复制混淆矩阵和检测效果图到仓库 results/
Copy-Item "D:\YOLO_Archive\yolo26模型进化\V16.0\YOLO26_V16.0_confusion_matrix.png" "C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\aerial-yolo26s-drone-detection\results\confusion_matrix.png"
Copy-Item "D:\YOLO_Archive\yolo26模型进化\V16.0\YOLO26_V16.0_images\val_batch0_pred.jpg" "C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\aerial-yolo26s-drone-detection\results\"
Copy-Item "D:\YOLO_Archive\yolo26模型进化\V16.0\YOLO26_V16.0_images\val_batch1_pred.jpg" "C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\aerial-yolo26s-drone-detection\results\"
```

### 验证消融表数据

```powershell
# V15 最终 mAP50-95
Select-String -Path "D:\YOLO_Archive\yolo26模型进化\V15.0\YOLO26_V15.0_results.csv" -Pattern '^151,' | ForEach-Object { $_.Line.Split(',')[8] }

# 同理可查 V17/V18/V19
```

### 推送更新

```bash
cd "C:/Users/ZHY/WorkBuddy/2026-07-28-18-27-55/aerial-yolo26s-drone-detection"
git add -A
git commit -m "Add results images, technical report, and verified metrics"
git push
```

---

## 九、关联文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| 版本总览 | `D:\YOLO_Archive\yolo26模型进化\版本总览.md` | 20+ 版本消融记录 |
| 训练记录 | `D:\YOLO_Archive\yolo26模型进化\V*/训练记录.md` | 每个版本的详细过程 |
| V16 AI 文档 | `D:\YOLO_Archive\yolo26模型进化\V16.0\YOLO26_V16.0_AI文档.md` | V16 架构与收敛分析 |
| GitHub 指南 | `C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\YOLO项目上GitHub指南.md` | 完整建仓、推送、自查流程 |
| README 骨架 | `C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\GitHub_README骨架.md` | 仓库结构模板 |
| 当前 README | `C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\aerial-yolo26s-drone-detection\README.md` | 已部署的 README |
| 申请材料 | `C:\Users\ZHY\WorkBuddy\2026-07-28-18-27-55\` | SOP/CV/推荐信等 |

---

> **一句话总结**：帮邹昊宜检查并完善 GitHub 仓库 `Drone_yolo_7types`，补配图、验证数据、加技术报告，确保招生官能在 3 分钟内看懂这个项目的研究价值。所有数据必须真实，已在第三节完整列出。
