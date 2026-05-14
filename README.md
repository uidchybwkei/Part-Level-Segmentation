# Semantic-SAM 环境配置记录：Python 3.11.9 + PyTorch 2.8.0 + CUDA 12.9

本文记录本项目在以下环境中的成功配置方式：

- **Python**: 3.11.9
- **PyTorch**: 2.8.0+cu129
- **TorchVision**: 0.23.0+cu129
- **CUDA runtime/toolkit**: 12.9
- **cuDNN**: 9.10.2
- **GPU**: NVIDIA A800-SXM4-80GB

## 1. 创建独立 conda 环境

```bash
conda create -n semsam-py311-cu129 python=3.11.9 -y
conda activate semsam-py311-cu129
python -m pip install --upgrade pip setuptools wheel
```

## 2. 安装 PyTorch 2.8.0 cu129

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
```

验证：

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()); print(torchvision.__version__); print(torch.cuda.is_available())"
```

成功时应看到类似：

```text
2.8.0+cu129 12.9 91002
0.23.0+cu129
True
```

## 3. 安装 CUDA 12.9 编译工具链

Semantic-SAM 依赖 `detectron2-xyz` 和 `MultiScaleDeformableAttention` CUDA 扩展，需要完整 `nvcc`。

```bash
conda install -c nvidia/label/cuda-12.9.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev -y
```

验证：

```bash
nvcc --version
```

成功时应看到 `release 12.9, V12.9.86`。

## 4. 安装项目 Python 依赖

在 `Semantic-SAM` 目录下执行：

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/panopticapi.git'
```

本项目建议保留：

- **numpy<2**: 避免旧代码和部分 COCO 工具链的兼容问题。
- **pillow<10**: 兼容 `detectron2-xyz` 中仍使用的旧 Pillow 常量。
- **scikit-image<0.22**: 与 `pillow<10` 搭配更稳定。
- **iopath<0.1.10**: 与 `detectron2==0.6` 的依赖约束一致。

## 5. 编译安装 detectron2-xyz

本仓库同级目录已有 `detectron2-xyz`，推荐使用本地 editable 安装：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
pip install --no-build-isolation -e ../detectron2-xyz
```

说明：

- **CUDA_HOME** 指向当前 conda 环境，确保使用 CUDA 12.9 的 `nvcc`。
- **TORCH_CUDA_ARCH_LIST=8.0** 对应 A800/A100 这类 Ampere GPU。
- **MAX_JOBS=4** 可以降低编译时 CPU/内存压力。

## 6. 编译 Semantic-SAM CUDA op

```bash
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
cd semantic_sam/body/encoder/ops
python setup.py build install
cd -
```

## 7. 核心导入验证

在 `Semantic-SAM` 目录下执行：

```bash
python -c "import torch, torchvision, detectron2, MultiScaleDeformableAttention; from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator; print('ok', torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torchvision.__version__, detectron2.__version__)"
```

成功输出示例：

```text
ok 2.8.0+cu129 12.9 91002 0.23.0+cu129 0.6
```

## 8. 下载模型权重

在 `Semantic-SAM` 目录下下载模型权重：

```bash
wget https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
```

或者手动下载后放置到 `Semantic-SAM` 目录下。

## 9. 启动 FastAPI WebUI

确认 `swinl_only_sam_many2many.pth` 位于 `Semantic-SAM` 目录下，然后执行：

```bash
python fastapi_webui.py --ckpt swinl_only_sam_many2many.pth --model_type L --port 7860
```

**命令行参数默认值：**
- `--ckpt`: `swinl_only_sam_many2many.pth`
- `--model_type`: `L` (可选 `T`)
- `--port`: `7860`

**WebUI API 参数默认值：**
- `level_choice`: `All (1-6)` (使用所有粒度提示)
- `points_per_side`: `64` (网格点密度)
- `pred_iou_thresh`: `0.80` (预测 IoU 阈值)
- `stability_score_thresh`: `0.90` (稳定性分数阈值)
- `postprocess`: `small_first_cut` (后处理方式，可选 `hierarchy_cut` 或 `none`)
- `min_mask_area`: `1` (最小 mask 面积)
- `min_visible_ratio`: `0.01` (最小可见比例)
- `contain_thresh`: `0.80` (包含阈值，仅 hierarchy_cut 使用)

### Python 函数调用说明

如果不使用本项目自带 WebUI，而是自己写 endpoint，可以直接调用 Semantic-SAM 的模型构建函数和自动 mask 生成器。

**核心调用链：**
- `build_semantic_sam(model_type="L", ckpt="swinl_only_sam_many2many.pth")`: 加载 Semantic-SAM 模型。
- `prepare_image_from_pil(pil_image, short_edge=640)`: 将 PIL 图片转换为模型输入张量。
- `SemanticSamAutomaticMaskGenerator(...)`: 创建自动分割生成器。
- `mask_generator.generate(images)`: 对输入图片生成 masks。
- `small_first_cut_masks(...)` / `hierarchy_cut_masks(...)`: 可选后处理，用于裁剪重叠 masks。

**模型加载参数：**
- `model_type`: 默认 `L`。可选 `L` 或 `T`，需要与权重文件对应。
- `ckpt`: 默认 `swinl_only_sam_many2many.pth`。模型权重路径。

**图片预处理参数：**
- `pil_image`: PIL RGB 图片。
- `short_edge`: 默认 `640`。输入图片短边缩放尺寸。

**SemanticSamAutomaticMaskGenerator 参数：**
- `model`: `build_semantic_sam` 返回的模型对象。
- `points_per_side`: 默认 `64`。网格点密度，数值越大 mask 可能越多，但推理更慢、显存占用更高。
- `points_per_batch`: 当前默认 `64`。
- `pred_iou_thresh`: 默认 `0.80`。预测 IoU 阈值。
- `stability_score_thresh`: 默认 `0.90`。稳定性分数阈值。
- `level`: 默认 `[1, 2, 3, 4, 5, 6]`。粒度提示，`[1, 2]` 偏 semantic level，`[3]` 偏 instance level，`[4, 5, 6]` 偏 part level。

**后处理参数：**
- `postprocess`: 默认 `small_first_cut`。可选 `small_first_cut`、`hierarchy_cut`、`none`。
- `min_mask_area`: 默认 `1`。后处理时保留的最小 mask 面积。
- `min_visible_ratio`: 默认 `0.01`。后处理时保留的最小可见比例。
- `contain_thresh`: 默认 `0.80`。仅 `hierarchy_cut` 使用，用于判断父子 mask 包含关系。

**最小 Python 调用示例：**

```python
import torch
from PIL import Image

from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
from semantic_sam.inference_utils import prepare_image_from_pil
from semantic_sam.mask_postprocess import small_first_cut_masks

model = build_semantic_sam(model_type="L", ckpt="swinl_only_sam_many2many.pth")

pil_image = Image.open("examples/dog.jpg").convert("RGB")
image_ori, images = prepare_image_from_pil(pil_image, short_edge=640)

mask_generator = SemanticSamAutomaticMaskGenerator(
    model,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.90,
    level=[1, 2, 3, 4, 5, 6],
)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    masks = mask_generator.generate(images)

masks = small_first_cut_masks(
    masks,
    min_mask_area=1,
    min_visible_ratio=0.01,
)
```

`masks` 是一个 list，每个元素包含 `segmentation`、`area`、`predicted_iou`、`stability_score` 等信息，可以按自己的业务需求转成 JSON、RLE、PNG mask 或 SVG。

成功时日志应包含：

```text
[FastAPI WebUI] Loading model type=L, ckpt=swinl_only_sam_many2many.pth
[FastAPI WebUI] Model loaded
Uvicorn running on http://0.0.0.0:7860
```

本次验证中，WebUI 成功启动，首页请求返回 `200 OK`，`/api/auto` 自动分割接口也返回 `200 OK`。
