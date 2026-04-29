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

## 8. 启动 FastAPI WebUI

确认 `swinl_only_sam_many2many.pth` 位于 `Semantic-SAM` 目录下，然后执行：

```bash
python fastapi_webui.py --ckpt swinl_only_sam_many2many.pth --model_type L --port 7860
```

成功时日志应包含：

```text
[FastAPI WebUI] Loading model type=L, ckpt=swinl_only_sam_many2many.pth
[FastAPI WebUI] Model loaded
Uvicorn running on http://0.0.0.0:7860
```

本次验证中，WebUI 成功启动，首页请求返回 `200 OK`，`/api/auto` 自动分割接口也返回 `200 OK`。
