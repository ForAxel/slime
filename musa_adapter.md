# Slime MUSA 适配操作指南

本文总结在 Slime 中把 CUDA 优先的训练/推理链路适配到 MUSA 时采用的方法。它是一份可复用的实施与审查指南；当前分支的准确提交边界、文件改动和运行限制见 [musify.md](musify.md)。

## 1. 适配目标与原则

MUSA 适配不是机械替换 `cuda`、`nccl` 等字符串，而是在保留 CUDA 默认行为的同时，消除通用运行路径对 CUDA 设备、NCCL、可见设备和 CUDA 专有 API 的硬编码。

实施时遵循以下原则：

1. 先固定官方基线、适配提交和临时补丁边界，再讨论当前能力。
2. 通用设备操作收敛到一个后端抽象层，业务代码不散落 `if MUSA`。
3. `torch_musa`、`musa_patch`、Megatron、SGLang 和 Slime 各自负责的兼容范围必须分清。
4. accelerator bootstrap 必须在 Megatron/SGLang 及其间接导入之前实际执行，不能只看同一文件中的文本位置。
5. 通信后端和权重同步协议分别验证；`NCCL -> MCCL` 不代表收发协议天然兼容。
6. CUDA-only kernel、离线量化工具和专有优化必须明确保留边界，不能宣称已由 API 改名获得 MUSA 支持。
7. 正式适配、CI 修复、环境临时 patch 和代码备份分开记录，避免把可运行 workaround 固化成公共设计。

## 2. 先固定提交边界

以当前 `dev/fb42ae4-musa-patch` 为例，审查应以如下线性历史为准：

| 层次 | 提交 | 职责 |
| --- | --- | --- |
| 官方基线 | `41014d1f29e201137fdffce737bb8bac65bc5219` | 当前 Slime 官方 `main`；其提交 ID 保持不变 |
| 核心适配 | `434549f669f1f2cfcd12242fc0afe020f1801ff0`（原 `6a830964`） | accelerator、运行时设备迁移、MCCL/Ray/权重更新以及基础测试 |
| CI 修复 | `c00cbfd68c80fb692c13ecc616993ad3f8414c5f`（原 `902b59ac`） | 使权重更新测试不依赖已安装的 Megatron，并补齐测试桩 |
| 临时 patch | `f9a662bea19d25ceaa302e81ac15cb793b3fe0ab`（原 `200a84b`） | MUSA 环境运行所需的通信模式、参数兼容和后端保护 workaround |
| 备份提交 | `WIP: tmp commit to backup code` | 文档、示例运行辅助、reward 兼容等备份内容，不属于核心 MUSA 抽象 |

推荐每次先执行：

```bash
git branch --show-current
git log --oneline --decorate --graph 41014d1..HEAD
git show --stat <commit>
git diff --name-status <commit>^ <commit>
```

不能只看 `41014d1..HEAD` 的总差异：这样会把 `zero2one` reward、router 兼容、`.rayignore` 等备份内容误写成 MUSA 核心能力。

## 3. 调查 CUDA 假设

搜索范围至少覆盖设备、内存、通信、Ray、profiler、编译工具和模型插件：

```bash
rg -n "torch\.cuda|\.cuda\(|device=['\"]cuda|nccl|CUDA_VISIBLE_DEVICES|_cuda|is_cuda|get_device_capability" \
  slime slime_plugins tools
```

将结果分为三类：

- **通用运行路径**：设备构造、tensor 搬运、同步、cache、stream、显存统计和进程组，应通过 accelerator 适配。
- **条件兼容路径**：MCCL、Ray 可见设备、MUSA profiler 私有接口、SGLang 版本差异，需要能力检测或显式分支。
- **CUDA-only 边界**：DeepGEMM、确定性路由 CUDA kernel、FlashQLA/SM90、INT4 CUDA 扩展和 CUDA 离线量化工具，应保留限制或增加明确的不支持错误。

每个命中都要回答：它是否位于真实 MUSA 运行链路、是否存在语义等价 API、是否已经由外部 patch 负责、是否有测试证明。

## 4. 设计 accelerator 层

当前 Slime 使用 `slime/utils/accelerator/` 包：

- `base.py` 定义 Slime 实际使用的轻量公共契约，包括设备、stream/event、AMP、RNG、内存、通信和能力查询；
- `torch_accelerator.py` 复用 CUDA-like PyTorch namespace 的公共转发逻辑；
- `cuda.py`、`musa.py` 分别收敛后端语义和厂商专属 API；
- `__init__.py` 负责注册、选择、线程安全单例和旧模块级 API 兼容。

选择规则为：

1. `SLIME_ACCELERATOR=<backend>` 是最高优先级显式覆盖，内置支持 `cuda`、`musa`，也接受已注册的第三方后端；
2. `MUSA_VISIBLE_DEVICES` 或 `MUSA_PATCH_PATH` 继续作为兼容的显式 MUSA 请求；
3. 自动检测只接受真实 `is_available()`，按注册 priority、同 priority 名称排序确定结果；内置顺序为 MUSA、CUDA；
4. MUSA 不可用时会检测 CUDA；CUDA 也不可用时立即报错，不伪造 CUDA 可用性，也不回落到 CPU；
5. 显式指定不可用或未知后端时立即报错，不回落到其他设备；选择结果在进程内稳定缓存。

新增后端只需继承 `Accelerator`（或复用 `TorchAccelerator`）并在第一次选择前调用：

```python
accelerator.register_accelerator(
    "npu",
    NPUAccelerator,
    npu_is_available,
    priority=150,
    communication_backends=("hccl",),
)
```

核心业务无需增加 `if npu`。可选特性通过 `supports(<capability>)` 查询；不支持的 stream、event 或 AMP 等接口应返回明确的 `False` 或抛出 `NotImplementedError`，不能伪造成功。

核心映射为：

| 原 CUDA 假设 | accelerator API | MUSA 行为 |
| --- | --- | --- |
| `torch.cuda.current_device()` | `current_device()` / `device()` | 当前 MUSA 设备 |
| `torch.cuda.set_device(x)` | `set_device(x)` | `torch.musa.set_device(x)` |
| `.cuda()` / `device="cuda"` | `.to(device=accelerator.device())` | 运行时选择 MUSA |
| `torch.cuda.synchronize()` | `synchronize()` | MUSA 同步 |
| cache / IPC / memory / stream | 对应 accelerator 方法 | 转发至 `torch.musa` |
| 默认 `nccl` | `process_group_backend()` | MUSA 下返回 `mccl` |
| 权重更新 `nccl` | `weight_update_backend()` | MUSA 下返回 `cpu:gloo,musa:mccl` |

抽象层的约束：

- 未请求或未检测到 MUSA 时，在 CUDA 真实可用的前提下保持 CUDA 原行为；
- 没有可用加速器时立即报错；CPU 仅可作为测试 fake 的 tensor 存储设备，不是正式后端；
- 设置了 `MUSA_VISIBLE_DEVICES` 或 `MUSA_PATCH_PATH`、但 `torch.musa` 不可用时，通信后端选择必须显式报错，不能静默回落到 NCCL；
- 没有等价语义的专有 API 不做伪实现；
- 业务调用方直接使用公共 accelerator API，不再增加只做转发的薄 wrapper。

## 5. Bootstrap 与导入顺序

`musa_patch` 可能在导入时安装 PyTorch/Megatron 兼容逻辑，因此所有独立入口都应满足：

```python
from slime.utils import accelerator

# isort: split

import torch
from megatron.core import ...
```

`# isort: split` 是执行顺序保护，不是装饰。修改 import 后必须重新检查 formatter 没有把 accelerator 移到 Megatron/SGLang 后面。

正式 bootstrap 的职责是：

1. 在 `SLIME_ACCELERATOR=musa`、`MUSA_VISIBLE_DEVICES`、`MUSA_PATCH_PATH` 明确请求，或 `torch.musa.is_available()` 自动检测成功时提前 bootstrap；
2. 将 `MUSA_PATCH_PATH` 加入 `sys.path`；
3. 单次导入 `musa_patch`；
4. 不在普通 CPU/CUDA 环境导入 `torch_musa` 或 `musa_patch`；
5. patch 自身依赖缺失或初始化异常时保留根因并给出明确错误。

业务入口不再直接导入 `musa_patch`。确需在第三方 torch 模块加载后补 patch 时，统一调用 `accelerator.post_import_torch()`；CUDA 为无操作，MUSA 后端负责厂商逻辑。

## 6. 分布式、Ray 与权重同步

### 6.1 进程组

- 训练默认后端：CUDA 使用 NCCL，MUSA 使用 MCCL。
- 权重同步组：MUSA 使用 `cpu:gloo,musa:mccl`，同时承载 CPU 元数据和 MUSA tensor。
- WORLD/子进程组销毁重建：识别 `nccl`、`mccl` 和复合 MUSA backend，保存实际 rendezvous/backend 状态后再重建。
- 不能把字符串映射当成集成验证；MCCL collective 仍需真实多机测试。

### 6.2 Ray 设备映射

- 根据运行环境选择 `CUDA_VISIBLE_DEVICES` 或 `MUSA_VISIBLE_DEVICES`；
- 同时接受 Ray 返回的物理 ID 和局部 ID，并在非法映射时显示当前环境变量；
- 支持 `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES`；
- MUSA 环境跳过 NVIDIA NVML NUMA affinity；设备没有 NVIDIA UUID 时回退到设备序号。

### 6.3 权重同步协议

后端和协议必须分开描述。核心适配默认保留异步 broadcast；临时 patch 通过 `UPDATE_MODE` 提供：

- `broadcast`：设备组异步 broadcast；
- `sync-broadcast`：设备组同步 broadcast；
- `gloo-broadcast`：CPU tensor 同步 broadcast；
- `p2p-broadcast`：训练 rank 0 逐 tensor 发送给组内 rank 1，再由接收端按约定处理。

验证时至少覆盖元数据字段、rank 映射、tag 顺序、空 bucket、连续多次更新及多 PP/TP/engine 拓扑。临时环境变量协议在进入正式实现前，应改为显式参数并与 SGLang receiver 做版本绑定。

## 7. 临时 patch 的管理方式

临时 patch 可以帮助当前环境跑通，但必须在文档和提交历史中满足：

- 明确触发的实际报错、依赖版本和适用环境；
- 不反向宣称为通用 MUSA 能力；
- 不改变普通 CUDA 默认行为，或明确记录行为差异；
- 有退出条件：依赖升级、上游修复或正式参数化后删除；
- 与 reward、router、示例打包等非设备适配内容分离。

当前 `f9a662b` 就属于这一层；末尾的 `WIP: tmp commit to backup code` 是备份层。后续向官方提交时，应以 `434549f + c00cbfd` 为核心审查对象，再逐项决定是否吸收临时 patch。

## 8. 测试与验证

无加速器硬件的测试通过独立 `FakeAccelerator` 验证纯逻辑和导入边界，fake 不是正式 CPU 后端：

- 可见设备 ID 映射；
- NCCL/MCCL/复合 backend 选择；
- MUSA 被请求但运行时不可用时的错误；
- `musa_patch` 单次导入及 CUDA 环境不导入；
- 显式 `SLIME_ACCELERATOR` 覆盖、未知后端和不可用后端错误；
- 多候选后端的确定性选择、公共 API 转发和第三方注册扩展；
- mock CUDA/MUSA 的设备、同步、内存和通信关键接口；
- reloadable process group 的 backend 识别和生命周期；
- 权重同步协议与空 colocated bucket。

测试应通过最小模块桩隔离 Megatron/Ray 等重量依赖，不能用 `importorskip("megatron")` 把核心断言跳过。CI 模板和生成后的 workflow 必须同步。

推荐检查：

```bash
PYTHONPATH=. pytest -q \
  tests/test_accelerator.py \
  tests/test_reloadable_process_group_world.py \
  tests/test_update_weight_transport.py \
  tests/test_empty_colocated_weight_bucket.py
python -m compileall -q slime/utils/accelerator
pre-commit run --files <changed-python-files>
git diff --check
```

无硬件测试通过不等于 MUSA 集成通过。最终还需真实 MUSA 环境验证：Megatron 初始化、MCCL 多 rank、Ray 设备映射、SGLang rollout、连续权重更新、训练 step、checkpoint、profiler/OOM 路径。

## 9. 交付检查清单

- [ ] 官方基线、核心适配、CI 修复、临时 patch 和备份提交边界准确。
- [ ] accelerator 在 Megatron/SGLang 之前实际执行，且格式化后顺序仍正确。
- [ ] 普通 CUDA 环境不会导入 `musa_patch`。
- [ ] 没有把临时的重复 patch 调用描述为推荐设计。
- [ ] 设备、同步、cache、IPC、memory、stream 和 profiler 已逐类核查。
- [ ] NCCL/MCCL、复合 backend、Ray 可见设备和进程组重建已覆盖。
- [ ] 权重同步后端、协议和 SGLang receiver 约定分别验证。
- [ ] CUDA-only kernel/工具被明确列为限制。
- [ ] 无硬件测试通过独立 fake 真正执行，而不是因缺依赖 skipped。
- [ ] 已记录真实 MUSA 集群尚未覆盖的验证项。
- [ ] 对外文档不把 reward、router 或备份辅助文件算作核心 MUSA 适配。
