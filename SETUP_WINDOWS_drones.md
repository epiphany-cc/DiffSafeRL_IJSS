# Windows 环境配置说明（conda 环境 `drones`）

本文件记录 `DiffSafeRL0828` 在 Windows 上运行所需的完整环境配置。
配置日期：2026-09-11。

---

## 1. 环境基本信息

| 项目 | 值 |
|---|---|
| 环境名 | `drones` |
| 环境路径 | `K:\conda\envs\drones` |
| Python | 3.10.21（Anaconda 发行版，MSVC v.1942 构建） |
| conda | 24.9.2（Anaconda3 2024.10-1，非管理员安装） |
| 编译器 | MSVC 14.44.35207（VS 2022 Build Tools） |
| Eigen | 3.4.0（conda-forge） |

**为什么用 Python 3.10**：项目 `pyproject.toml` 要求 `^3.10`，且仓库里遗留的
`.pyc` 是 `cpython-310`，说明原始环境就是 3.10。

---

## 2. 激活环境（重要）

```powershell
conda activate drones
```

### ⚠️ 必须激活，不能直接调用 python.exe

conda-forge 的 numpy/pybullet 依赖 MKL 等 DLL，这些 DLL 位于
`K:\conda\envs\drones\Library\bin`。**只有 `conda activate` 才会把这个目录加进 PATH**。
若直接运行 `K:\conda\envs\drones\python.exe`，第一次触发 BLAS 调用（如 `np.dot`）
就会以 `0xC06D007F`（延迟加载 DLL 失败）崩溃，而且没有任何 Python 报错。

如果不想用 `conda activate`，至少手动补上 PATH：

```powershell
$env:PATH = "K:\conda\envs\drones;K:\conda\envs\drones\Library\mingw-w64\bin;" +
            "K:\conda\envs\drones\Library\usr\bin;K:\conda\envs\drones\Library\bin;" +
            "K:\conda\envs\drones\Scripts;K:\conda\envs\drones\bin;" + $env:PATH
```

---

## 3. 依赖清单

### 3.1 项目声明的依赖（`pyproject.toml`）

| 包 | 版本 | 来源 |
|---|---|---|
| numpy | 1.26.4 | pip（⚠️ 必须 <2） |
| scipy | 1.15.3 | pip |
| matplotlib | 3.10.9 | pip |
| transforms3d | 0.4.2 | pip |
| pytest | 7.4.4 | pip |
| pybullet | 3.2.5 | **conda-forge**（见 §4） |
| gymnasium | 0.28.1 | pip（⚠️ 不可升级到 0.29+） |
| stable-baselines3 | 2.0.0 | pip（⚠️ 不可升级到 2.1+） |
| tensorboard | 2.15.1 | pip（项目锁定） |
| tqdm | 4.66.2 | pip（项目锁定） |
| trajectory_cpp | 0.0.1 | 本地编译（见 §5） |

### 3.2 代码实际使用但未声明的依赖

| 包 | 版本 | 用在哪 |
|---|---|---|
| torch | 2.5.1+cu121 | stable-baselines3；`run_train` 硬编码 `device="cuda"` |
| qpth | 0.0.18 | HOCBF QP 层 |
| quadprog | 0.1.13 | QP 求解 |
| pandas | 2.3.3 | 结果处理 |
| pillow | 12.3.0 | `BaseAviary.py` 图像处理 |
| setuptools | 80.10.2 | ⚠️ **必须 <81**（见 §6.1） |
| cmake / pybind11 / ninja | 4.4.3 / 3.1.0 / 1.13.2 | 编译 trajectory_cpp 用 |

### 3.3 版本约束的原因

- **numpy < 2**：gymnasium 0.28 / stable-baselines3 2.0 / qpth 均为 numpy 1.x 时代产物。
- **gymnasium 0.28.1**：stable-baselines3 2.1+ 会强制拉 gymnasium ≥0.29，API 有变化。
- **stable-baselines3 2.0.0**：与项目原始代码匹配。

---

## 4. pybullet 为什么从 conda-forge 装

PyPI 上 **pybullet 从未发布过现代 Python 的 Windows wheel**（只有 2017 年的
cp35/cp27）。`pip install pybullet` 在 Windows 上必然回退到源码编译。

conda-forge 提供预编译的 win-64 版本，版本号写作 `3.25`（即 3.2.5）：

```powershell
conda install -n drones -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge ^
  pybullet=3.25 "numpy>=1.24,<2"
```

**必须同时约束 numpy**，否则 conda 会把 numpy 升到 2.x，破坏整个 pip 栈。

---

## 5. trajectory_cpp：缺失的 C++ 子模块

### 5.1 背景

`trajectories/trajectory_generation/` 原本是一个 **git 子模块**，指向
`https://github.com/danielbinschmid/trajectory_generation`（分支 `no_ros_dep`）。
拿到的代码包里该目录是空的，导致 `import trajectories` 直接失败。

现已从上游取回全部源码（`CMakeLists.txt`、`setup.py`、`src/`、`include/`、`python_lib/`）。

### 5.2 原版构建脚本为什么在 Windows 上不可用

| 问题 | 说明 |
|---|---|
| GCC 专用 flag | `-std=c++11`、`-O3 -fPIC` 在 MSVC 下无效 |
| pybind11 子模块为空 | `add_subdirectory(pybind11)` 找不到目录 |
| CMake 4 拒绝旧版本声明 | `cmake_minimum_required(VERSION 3.4...3.18)` 中 3.4 < 3.5 |
| MSVC 未定义 `M_PI` | `include/lib/root_finder.hpp` 使用了 `M_PI`，MSVC 默认不提供 |
| Windows 输出目录 | CMake 的 MODULE 库在 Windows 走 RUNTIME 输出目录 |

### 5.3 本项目的处理方式

- 原文件已备份为 **`setup.py.orig`** 和 **`CMakeLists.txt.orig`**（可随时还原）。
- 新的 `setup.py` 改用 pybind11 的 `Pybind11Extension` **直接编译**，完全绕开 CMake：
  - 自动定位 Eigen（支持 conda 的 `<prefix>/Library/include/eigen3`，也支持 `EIGEN_INCLUDE_DIR` 环境变量）
  - 通过 `_USE_MATH_DEFINES` 解决 MSVC 的 `M_PI` 问题
  - 编译 `src/pywrap.cpp`、`src/traj_gen.cpp`、`src/trajectory_wrapper.cpp`
- 新增 `pyproject.toml` 声明构建依赖，使 pip 的构建隔离环境能拿到 pybind11。

### 5.4 重新编译

```powershell
conda activate drones
cd "J:\epiphany\phd-paper\DiffSafeRL(third-paper-phd)\code\DiffSafeRL0828"
pip install -e .
```

### 5.5 ⚠️ 必须传 C 连续数组

`trajectory_cpp.calc_trajectory` 内部用 `Eigen::Map<RowMajor>` 直接读取 numpy 的原始
内存缓冲区，**不做布局检查**。因此传入的 `(3, N)` 数组必须是 **C 连续**的：

```python
wp = np.zeros((3, n))      # ✅ C 连续
wp = some_4x3_array.T      # ❌ 转置后是 Fortran 序，会算出完全错误的结果
```

项目自身代码（`polynomial_discretized_trajectory.py`）是正确的，此处仅提醒手工调用时注意。

---

## 6. 已知问题与修复记录

### 6.1 `pkg_resources` 缺失

`gym_pybullet_drones/envs/BaseAviary.py:7` 有 `import pkg_resources`。
setuptools ≥ 81 已移除该模块，而 conda 自带的 setuptools 是 83.0.0。

**修复**：`pip install "setuptools<81"`（当前为 80.10.2）

### 6.2 `0xC06D007F` 崩溃且无任何报错

见 §2：未激活环境导致 MKL DLL 找不到。

### 6.3 conda 命令在受限沙箱下崩溃

若在受限环境执行 conda 命令遇到 `PermissionError: [WinError 5]`（来自
`conda/plugins/virtual_packages/cuda.py` 的 `multiprocessing.SimpleQueue`），
设置环境变量即可跳过 CUDA 探测：

```powershell
$env:CONDA_OVERRIDE_CUDA='12.0'
```

普通 CMD/PowerShell 中不会有此问题。

### 6.4 `import torch` 报 `WinError 182`（fbgemm.dll 加载失败）

**症状**：

```
OSError: [WinError 182] 操作系统无法运行 %1。
Error loading "...\torch\lib\fbgemm.dll" or one of its dependencies.
```

**根因**：env 的 `Library\bin\libiomp5md.dll` 被 **`llvm-openmp` 用自己的 LLVM OpenMP
运行时覆盖**了。两个包的 DLL 大小完全相同（684,872 字节）：

```
Library\bin\libiomp5md.dll   684,872 字节   (LLVM 版，错误)
Library\bin\libomp.dll       684,872 字节   (同一个文件)
Library\bin\libiomp5md.pdb 4,476,928 字节   (Intel 的 pdb 未被覆盖，暴露了问题)
```

`fbgemm.dll` 是针对 **Intel OpenMP** 编译的，LLVM 版的导出序号对不上，
于是报 `WinError 182`（无效的序号）。

**修复步骤**：

```powershell
# 1. 移除 llvm-openmp（它没有其他依赖者）
conda remove -n drones llvm-openmp -y

# 2. 注意：上一步会把 libiomp5md.dll 一并删掉（conda 把该文件算在 llvm-openmp 名下）
#    强制重装 intel-openmp 以恢复文件并修正 conda 的归属记录
conda install -n drones --force-reinstall --no-deps "intel-openmp=2023.1.0" -y

# 3. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**正确的 DLL** 应为 **2,047,000 字节，文件版本 20230118**。检查方法：

```powershell
(Get-Item "K:\conda\envs\drones\Library\bin\libiomp5md.dll").VersionInfo.FileVersion
```

若将来再遇到同样报错，从 base 环境复制即可临时恢复：

```powershell
Copy-Item "K:\conda\Library\bin\libiomp5md.dll" "K:\conda\envs\drones\Library\bin\" -Force
```

---

## 7. 运行方式

```powershell
conda activate drones
cd "J:\epiphany\phd-paper\DiffSafeRL(third-paper-phd)\code\DiffSafeRL0828\agents"
python main.py --train True --timesteps 15000000 --n_envs 32 --gui False
```

注意 `agents/main.py` 中的关键参数：

- `TRAIN = True` / `TEST = False` / `VIS = False`
- `TIMESTEPS = 15000000`（完整训练量，调试时请调小）
- `EPISODE_LEN_SEC = 20`
- `pyb_freq = 240`，`ctrl_freq = 120`
- 模型保存路径：`../checkpointed_models/{时间戳}_model_for_DiffPPO-LSE`
- `train_policy.py` 中 `device="cuda"` 为硬编码，需要 CUDA 版 torch

---

## 8. 验证结果

配置完成后通过了以下测试：

| 测试 | 结果 |
|---|---|
| 16 个 Python 依赖导入 | ✅ 全部通过 |
| `import trajectories`（原报错点） | ✅ 通过 |
| 圆形 / 螺旋 / 随机 / 直线轨迹生成 | ✅ 通过（走 C++ 后端） |
| `trajectory_cpp.calc_trajectory` 数值正确性 | ✅ 起止点精确、t=1.0 处精确穿过航点 |
| `UZHAviary` 环境创建 / reset / 连续 30 步 | ✅ 通过（obs 形状 `(1,52)`，action `(1,8)`） |
| `DummyVecEnv` 训练环境 reset / step | ✅ 通过 |
| `pip install -e .` | ✅ 通过 |
| GPU 可用性（`torch.cuda.is_available()` + 真实 GPU 矩阵乘） | ✅ 通过 |

---

## 9. CUDA 版 torch 安装记录

### 9.1 硬件与选型

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（Ampere，sm_86，6 GB） |
| 驱动 | 546.80（支持 CUDA 12.3） |
| 选定版本 | **pytorch 2.5.1，CUDA 12.1，cuDNN 9** |

选 cuda12.1 而非 12.4：cu121 只要求驱动 ≥ 527.41，你的 546.80 完全满足，
不依赖 CUDA 12.x 的 minor version compatibility 机制，最稳。

### 9.2 安装命令

先卸掉 pip 版 CPU torch，避免两份 torch 打架：

```powershell
conda activate drones
pip uninstall -y torch
```

再从官方源安装。**注意必须同时给 `pytorch` 和 `nvidia` 两个 channel**：

```powershell
conda install --override-channels -c https://conda.anaconda.org/pytorch -c https://conda.anaconda.org/nvidia -c defaults "pytorch=2.5.1=py3.10_cuda12.1_cudnn9_0"
```

依赖链是：

```
pytorch 2.5.1 (cuda12.1)          <- channel: pytorch
   +- pytorch-cuda 12.1            <- channel: pytorch
   |    +- cuda-cudart 12.1        <- channel: nvidia   <-- 少了这个 channel 会解不出来
   +- mkl / intel-openmp           <- channel: defaults
```

Windows 上 `pytorch-cuda` **是存在的**（pytorch 2.5.x 起，Windows 也改用
`pytorch-cuda` + nvidia 运行时包的拆分结构）。网上很多"Windows 不需要
`pytorch-cuda`"的说法针对的是旧版本。

- `--override-channels` 不能省：`.condarc` 里配了清华镜像，不加会混源
- `-c defaults` 不能省：pytorch channel 不提供 `mkl`、`intel-openmp`

### 9.3 安装后必须做的两件事

1. **检查 numpy 没被升级**：conda 会把 numpy 换成 `defaults` 频道的构建，
   当前仍是 `1.26.4`。若变成 2.x，执行 `pip install "numpy<2"`。
2. **检查 `libiomp5md.dll`**：见 §6.4，这是装完 CUDA torch 后最容易踩的坑。

### 9.4 验证输出

```
torch version : 2.5.1
CUDA available: True
CUDA version  : 12.1
cuDNN version : 90100
device count  : 1
device name   : NVIDIA GeForce RTX 3060 Laptop GPU
compute cap   : sm_86
VRAM total    : 6.00 GB
GPU matmul OK : float
```

