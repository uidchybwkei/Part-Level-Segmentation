# Semantic-SAM @ Blackwell (RTX PRO 6000) 适配全过程

本文档记录了在 **NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96GB)** 上，
把官方 [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) 跑起来并优化到可用速度的完整过程。
原仓库的 `requirements.txt` / `INSTALL` 是基于 `torch 1.13 + cu113` 的，Blackwell 完全没法用
（最低需要 `torch ≥ 2.7 + cu128`），并且代码里有若干 PIL / PyTorch / NumPy 在新版本上的兼容性问题，
顺带还有两个非常严重的性能 bug。本文逐步给出所有改动。

---

## 0. 硬件与目标

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`, 97887 MiB) |
| 驱动 / CUDA | NVIDIA-SMI 590.44.01 / CUDA 13.1 runtime |
| OS | Ubuntu 20.04, Linux container（AutoDL） |
| 目标 | 让 `fastapi_webui.py` 能在该卡上加载 SwinL 检查点并跑 auto-mask 推理 |
| 工作目录 | `/root/autodl-tmp/Semantic-SAM` |
| 检查点 | `swinl_only_sam_many2many.pth`（SwinL 主干） |

```bash
nvidia-smi
# Driver 590.44.01, CUDA 13.1, RTX PRO 6000 Blackwell, 97887 MiB
```

---

## 1. 为什么不能用原 requirements

原 `requirements.txt` 头几行：

```text
torch
torchvision
... timm==0.4.12, numpy==1.23.5, transformers==4.19.2, kornia==0.6.4, gradio==3.35.2, pillow==9.4.0
```

并且 README 让你装：

```bash
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

在 Blackwell 上这套**完全不能跑**，原因：

1. **架构不支持**：Blackwell 是 `sm_120`，官方 PyTorch 从 **2.7 + cu128** 开始才在 wheel 里带 `sm_120` 内核；任何 `cu113 / cu117 / cu118` 的 wheel 在这块卡上都会立刻 `CUDA error: no kernel image is available for execution on the device`。
2. **Python 版本**：`torch ≥ 2.7` 要求 `Python ≥ 3.9`，本机 base 环境是 Python 3.8（原作者就是用 3.8 + torch 1.11 装的），必须新建 conda 环境。
3. **依赖钉死的旧版本与新工具链冲突**：`transformers==4.19.2`、`pillow==9.4.0`、`gradio==3.35.2` 等在新栈上要么装不上要么和别的包冲突。
4. **detectron2-xyz**：Semantic-SAM 依赖 `MaureenZOU/detectron2-xyz`（一个 detectron2 fork），这个 fork 是按 torch 1.x 编的，必须**重新对着 torch 2.7 重编**。
5. **MultiScaleDeformableAttention CUDA op**：`semantic_sam/body/encoder/ops` 下有自带的 CUDA 扩展，需要本机 `nvcc 12.x` 才能编出 `sm_120` 的 binary，并且其源码用了 torch 1.x 的废弃 API。

---

## 2. 整体方案

```text
新建 conda 环境 (Python 3.10)
   └── 安装 CUDA 12.8.1 toolkit (nvcc / cudart-dev / libraries-dev)
   └── 安装 torch 2.7.1 + torchvision 0.22.1 (cu128 wheel, 自带 sm_120 kernel)
   └── 安装并重编 detectron2-xyz (FORCE_CUDA=1, sm_120)
   └── 安装 panopticapi + 其他 Python 依赖（numpy<2, pillow<10, scikit-image<0.22, transformers 4.46.3 ...）
   └── 重编 MultiScaleDeformableAttention（要先打两处 patch）
   └── 修源码里 PIL / NumPy / 性能相关的 4 处兼容性问题
   └── 启动 fastapi_webui.py
```

---

## 3. 创建 Python 3.10 conda 环境

```bash
/root/miniconda3/bin/conda create -n semsam python=3.10 -y
source /root/miniconda3/bin/activate semsam   # 之后所有命令都在该环境里

python -m pip install --upgrade pip wheel setuptools ninja
```

**为什么 3.10**：3.10 是 torch 2.7 + cu128 wheel 同时支持的最稳的最低线，
3.11 / 3.12 也能跑，但有些老依赖（`timm==0.4.12` 之类）在 3.11+ 上偶有兼容问题。

---

## 4. 安装 CUDA 12.8 工具链（nvcc）

系统自带的 `nvcc` 是 11.3，绝对不能用来编 `sm_120`。直接装到环境里：

```bash
conda install -n semsam -c "nvidia/label/cuda-12.8.1" \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev -y
```

验证：

```bash
$CONDA_PREFIX/bin/nvcc --version
# Cuda compilation tools, release 12.8, V12.8.93
```

> 注意：conda 的 cuda 包把头文件放在 `$CONDA_PREFIX/targets/x86_64-linux/include/` 而不是 `$CONDA_PREFIX/include/`，
> nvcc 会找不到 `cuda_runtime_api.h`，**第 6 步会做软链修补**。

---

## 5. 安装 PyTorch 2.7.1 + cu128

```bash
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

验证 Blackwell 能跑：

```bash
python - <<'EOF'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(2, 3, device='cuda'); print((x @ x.T).cpu())
EOF
# 2.7.1+cu128 12.8 True
# NVIDIA RTX PRO 6000 Blackwell Server Edition (12, 0)
```

`(12, 0)` 即 `sm_120`，能跑就说明 wheel 内核 OK。

---

## 6. 修补 conda CUDA 头文件路径

让 `CUDA_HOME=$CONDA_PREFIX` 这一惯用约定能用：

```bash
# 把 targets/x86_64-linux/include/* 软链到 $CONDA_PREFIX/include/
TARGETS=$CONDA_PREFIX/targets/x86_64-linux/include
for f in $TARGETS/*.h $TARGETS/*.hpp; do
    ln -sfn "$f" "$CONDA_PREFIX/include/" 2>/dev/null
done
for d in $TARGETS/*/; do
    ln -sfn "$d" "$CONDA_PREFIX/include/" 2>/dev/null
done
ls $CONDA_PREFIX/include/cuda_runtime_api.h $CONDA_PREFIX/include/cuda.h
```

不做这一步，后面所有 CUDA 扩展编译都会以
`fatal error: cuda_runtime_api.h: No such file or directory` 失败。

---

## 7. 安装 detectron2-xyz

这是 Semantic-SAM 必需的 detectron2 fork（`from detectron2.modeling import ShapeSpec` 等）。
仓库已经 clone 在 `/root/autodl-tmp/detectron2-xyz`。**关键是要让 `nvcc` 编出 `sm_120`：**

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"   # 12.0 = Blackwell
export FORCE_CUDA=1

pip install -e /root/autodl-tmp/detectron2-xyz --no-build-isolation
```

`--no-build-isolation` 必须加，否则 pip 会在隔离环境里再装一份 torch（版本可能不对），
导致编出来的 `_C.so` 和我们运行环境的 torch ABI 不匹配。

**首次 import 时会爆的兼容性问题（已修）：**

`@/root/autodl-tmp/detectron2-xyz/detectron2/data/transforms/transform.py:46` 用了
`Image.LINEAR`，这个常量在 Pillow 10 被移除（改名 `BILINEAR`）。
所以下面**Pillow 必须钉死 `<10`**，scikit-image 也得跟着 `<0.22`。

---

## 8. 安装其他 Python 依赖

```bash
pip install \
    "numpy<2" opencv-python "pillow<10" scipy "scikit-image<0.22" matplotlib \
    einops shapely pycocotools \
    timm==0.4.12 kornia==0.6.12 torchmetrics fvcore \
    transformers==4.46.3 tokenizers safetensors huggingface-hub \
    sentencepiece ftfy regex \
    pyyaml yacs json_tricks omegaconf hydra-core iopath portalocker \
    tabulate cloudpickle progressbar2 vision-datasets==0.2.2 \
    fastapi "uvicorn[standard]" python-multipart

pip install "git+https://github.com/cocodataset/panopticapi.git"
```

钉版本的原因：

- `numpy<2`：`pycocotools` 等老调用方还在用 `np.float / np.int` 这种 NumPy 2 删掉的别名。
- `pillow<10`：`detectron2-xyz` 用 `PIL.Image.LINEAR`（见上）。
- `scikit-image<0.22`：scikit-image 0.22+ 强依赖 `pillow≥10.1`，会和上一行打架。
- `timm==0.4.12`：Semantic-SAM 的 SwinL backbone 直接用了 timm 0.4 的内部接口，新 timm 改名了。
- `kornia==0.6.12`：原仓库写的 0.6.4，但 0.6.4 装不上 py3.10；0.6.12 是同 minor 系列里能装的。
- `transformers==4.46.3`：原仓库 4.19.2 太老（py3.10 装不上），semantic-sam 只用了 `CLIPTokenizer / AutoTokenizer`，4.46.3 可以。

---

## 9. 编译 MultiScaleDeformableAttention CUDA 扩展

`semantic_sam/body/encoder/ops/` 下是 Deformable DETR 的自定义算子，
源码用了 torch 1.x 的废弃 API，先打 patch 再编。

### 9.1 patch：废弃 API `value.type()` → `value.scalar_type()`

文件：`@/root/autodl-tmp/Semantic-SAM/semantic_sam/body/encoder/ops/src/cuda/ms_deform_attn_cuda.cu`

torch 2.7 里 `AT_DISPATCH_FLOATING_TYPES` 第一个参数已不再接受
`at::DeprecatedTypeProperties`，必须传 `c10::ScalarType`：

```cpp
// 第 69 行
- AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda", ([&] {
+ AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {

// 第 139 行
- AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda", ([&] {
+ AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
```

### 9.2 编译

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
export FORCE_CUDA=1

cd /root/autodl-tmp/Semantic-SAM/semantic_sam/body/encoder/ops
rm -rf build
python setup.py build install
cd -
```

成功后 `import` 测试：

```bash
cd /root/autodl-tmp/Semantic-SAM
python -c "from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator; print('ok')"
# ok（中间会有几条 FutureWarning，无关紧要）
```

---

## 10. 跑通模型加载（基线）

```bash
cd /root/autodl-tmp/Semantic-SAM
python - <<'EOF'
from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
m = build_semantic_sam(model_type='L', ckpt='swinl_only_sam_many2many.pth')
print('model loaded ok')
EOF
```

到这一步，Semantic-SAM 已经"能跑"了。但 `fastapi_webui.py` 单图推理需要 ~65s，
和 4090 上的体验差 4 倍——下面是性能修复部分。

---

## 11. 性能修复（关键）

### 11.1 测出真正的瓶颈

按 batch 加 `torch.cuda.synchronize` 计时（同样的图 dog.jpg，
`points_per_side=64, level=[1..6]`，短边 640）：

```text
TOTAL: 69.07s  raw_masks=208
  model_forward       : 11.15s   ← 真正的 GPU 工作
  filter+stab+box+... :  1.03s
  rle  (mask_to_rle)  : 12.23s   ← CPU
  cat  (MaskData.cat) : 37.94s   ← CPU，且 O(N²)
```

GPU 实际只占 ~16%。**Pro 6000 不慢，是被 Python 里两段实现拖死了。**

### 11.2 修 `MaskData.cat` 的 O(N²) `deepcopy`

`@/root/autodl-tmp/Semantic-SAM/utils/sam_utils/amg.py:59-77`

原实现里每一个 batch 都对累计的 RLE 列表 `deepcopy(v)` + `list_a + list_b`，
RLE 是 `{"size":[h,w], "counts":[几千~几十万 ints]}`，64 个 batch 累积下来等于
每张图复制了 ~25k 个大字典，复杂度退化到 O(N²)。

新实现：浅引用 + `list.extend`，因为 `new_stats` 在 `cat` 之后就被丢弃，安全。

### 11.3 修 `mask_to_rle_pytorch` 的 per-mask 循环

`@/root/autodl-tmp/Semantic-SAM/utils/sam_utils/amg.py:114-166`

原实现对 24k 个 mask 逐个做布尔索引 `change_indices[change_indices[:,0]==i, 1]`
和单独的 `.cpu().tolist()`。改成：所有 GPU 索引一次算完 + `np.searchsorted` 切片 + 一次 `.cpu()`。
语义不变，输出格式与 pycocotools 兼容。

### 11.4 在 webui 里启用 cudnn benchmark / TF32

`@/root/autodl-tmp/Semantic-SAM/fastapi_webui.py:29-35`

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 11.5 修复后实测对比

同一张图、同一参数：

| 阶段 | 修复前 | 修复后 | 倍数 |
|---|---|---|---|
| `model_forward` | 11.15s | 11.77s | ≈1.0× （GPU 工作量没变） |
| `mask_to_rle` | 12.23s | **3.44s** | 3.5× |
| `MaskData.cat` | 37.94s | **0.03s** | ~1000× |
| **TOTAL** | **69.07s** | **23.13s** | **3.0×** |

输出 `raw_masks=208` 完全一致，没改语义。

---

## 12. 重写 requirements.txt

`@/root/autodl-tmp/Semantic-SAM/requirements.txt` 已重写，文件头注释里把上面所有
"必须分开手动装" 的步骤（torch cu128 / cuda toolkit / detectron2-xyz / panopticapi /
MSDeformAttn build）写清楚了，然后用一个干净的 pip 列表收尾。
别人在另一台 Blackwell 机器上 clone 完仓库照着 `requirements.txt` 头部走一遍即可复现。

---

## 13. 启动 FastAPI WebUI

```bash
conda activate semsam
cd /root/autodl-tmp/Semantic-SAM
nohup python fastapi_webui.py --port 7860 > /tmp/fastapi_webui.log 2>&1 &
tail -f /tmp/fastapi_webui.log
```

期待看到：

```text
[FastAPI WebUI] Loading model type=L, ckpt=swinl_only_sam_many2many.pth
[FastAPI WebUI] Model loaded
INFO:     Uvicorn running on http://0.0.0.0:7860
```

浏览器访问 `http://127.0.0.1:7860`。

---

## 14. 故障排查 cheatsheet

| 现象 | 原因 / 解法 |
|---|---|
| `CUDA error: no kernel image is available for execution on the device` | torch wheel 不带 `sm_120`，用 cu128 wheel：`pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128` |
| `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'` | Pillow ≥ 10，降到 `<10`。会顺带要求 `scikit-image<0.22`。 |
| `fatal error: cuda_runtime_api.h: No such file or directory` | conda 装的 cuda 头文件在 `$CUDA_HOME/targets/x86_64-linux/include/`，照第 6 节做软链。 |
| `sh: 1: cicc: not found` | 必须 `CUDA_HOME=$CONDA_PREFIX`（不是 `targets/x86_64-linux`），nvcc 才找得到 `nvvm/bin/cicc`。 |
| `subprocess.CalledProcessError: Command '['ninja', '-v']' returned 127` | `pip install ninja`，并确认 `$CONDA_PREFIX/bin` 在 `PATH` 里。 |
| `no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType"` | torch 2.x 废弃 API，按 9.1 节 patch `ms_deform_attn_cuda.cu`。 |
| 单图 60s+ 才出结果 | 没打 11.2 / 11.3 节的 patch；或者 webui 用了旧的 amg.py，重启进程即可。 |
| 模型加载完成但 `curl :7860` 拒绝 | uvicorn 还在启动，`tail -f /tmp/fastapi_webui.log` 等到 `Application startup complete`。 |

---

## 15. 关键改动清单（diff 全集）

代码改动只有三个文件：

1. `@/root/autodl-tmp/Semantic-SAM/semantic_sam/body/encoder/ops/src/cuda/ms_deform_attn_cuda.cu`
   - 第 69 行 `value.type()` → `value.scalar_type()`
   - 第 139 行 `value.type()` → `value.scalar_type()`
2. `@/root/autodl-tmp/Semantic-SAM/utils/sam_utils/amg.py`
   - `MaskData.cat`: 去掉 `deepcopy`，列表用 `extend`
   - `mask_to_rle_pytorch`: 去掉 per-mask Python loop，单次 `.cpu()` + `np.searchsorted`
3. `@/root/autodl-tmp/Semantic-SAM/fastapi_webui.py`
   - 启动时打开 `cudnn.benchmark` / TF32

外加一个外部仓库改动：`@/root/autodl-tmp/detectron2-xyz` 重新以
`TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"` 编了一遍（detectron2 自身的 C++ 代码
在 torch 2.7 + cu128 下 OK，无需 patch）。

最后是 `@/root/autodl-tmp/Semantic-SAM/requirements.txt`，被整体重写并加了详细的安装顺序注释。

---

## 16. 一键复现（在另一台 Blackwell 机器上）

```bash
# 0. 准备：clone 仓库（含 detectron2-xyz、Semantic-SAM 本身）
#    并把 swinl_only_sam_many2many.pth 放到 Semantic-SAM/ 目录下

# 1. conda 环境
conda create -n semsam python=3.10 -y && conda activate semsam
pip install -U pip wheel setuptools ninja

# 2. CUDA 12.8 toolkit
conda install -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev -y

# 3. 头文件软链（同 第 6 节）
TARGETS=$CONDA_PREFIX/targets/x86_64-linux/include
for f in $TARGETS/*.h $TARGETS/*.hpp; do ln -sfn "$f" "$CONDA_PREFIX/include/" 2>/dev/null; done
for d in $TARGETS/*/;        do ln -sfn "$d" "$CONDA_PREFIX/include/" 2>/dev/null; done

# 4. PyTorch (cu128 + sm_120)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 5. detectron2-xyz
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
export FORCE_CUDA=1
pip install -e /path/to/detectron2-xyz --no-build-isolation

# 6. 其它依赖
pip install -r /path/to/Semantic-SAM/requirements.txt
pip install "git+https://github.com/cocodataset/panopticapi.git"

# 7. 编 MSDeformAttn (确保第 9.1 节的 patch 已应用)
cd /path/to/Semantic-SAM/semantic_sam/body/encoder/ops
rm -rf build && python setup.py build install && cd -

# 8. 启动
cd /path/to/Semantic-SAM
nohup python fastapi_webui.py --port 7860 > /tmp/fastapi_webui.log 2>&1 &
tail -f /tmp/fastapi_webui.log
```
