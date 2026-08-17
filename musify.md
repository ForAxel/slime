# Slime MUSA 适配实现说明

本文描述 `dev/fb42ae4-musa-patch` 相对 Slime 官方基线的实际实现、提交边界、运行前提和已知限制。通用的适配方法与审查清单见 [musa_adapter.md](musa_adapter.md)。

## 1. 基线与当前提交链

- 官方基线：`41014d1f29e201137fdffce737bb8bac65bc5219`（`Add args check for --save-debug-train-data (#2276)`）。
- 当前分支：`dev/fb42ae4-musa-patch`。
- 核心 MUSA 适配由 `434549f`（原 `6a830964`）和 `c00cbfd`（原 `902b59ac`）组成；`f9a662b`（原 `200a84b`）是当前 MUSA 环境的临时运行 patch；末尾 `WIP` 提交是代码备份。

线性历史如下：

```text
681b3ad  旧 Slime 官方基线
└─ 2fa9a44  fix transform_ue8m0 in fp8 convert (#2271)
   └─ 876cd89  [ROCm] Support the INT4 QAT kernel on ROCm (#2274)
      └─ 00986d7  Fix model convert when use latest megatron (#2267)
         └─ 41014d1  当前 Slime 官方 main 基线
            └─ 434549f  feat: add backend-aware MUSA support
               └─ c00cbfd  fix(ci): make MUSA weight-update tests self-contained
                  └─ f9a662b  fix: isolate MUSA communication and backend guards
                     └─ WIP  tmp commit to backup code
```

提交职责必须分开理解：

| 提交 | 文件规模 | 定位 |
| --- | ---: | --- |
| `434549f`（原 `6a830964`） | 33 个文件，`+677/-131` | 正式的后端抽象、通用运行路径迁移、MCCL/Ray/权重同步适配，以及 latest-Megatron 参数入口 bootstrap |
| `c00cbfd`（原 `902b59ac`） | 3 个文件，`+78/-7` | CPU CI 自包含修复，不新增 MUSA 运行功能 |
| `f9a662b`（原 `200a84b`） | 8 个文件，`+67/-17` | 为当前依赖和集群运行追加的临时兼容/通信 patch |
| 末尾 `WIP` | 10 个文件（amend 后为准） | 文档、Ray 打包、router/reward 兼容和测试等备份内容 |

因此，对外说明核心适配时应以 `41014d1..c00cbfd` 为主；不能把 HEAD 总 diff 中的所有内容都归为 MUSA 设备适配。

### 1.1 本次纳入的官方 main 更新

官方 `681b3ad..41014d1` 的 4 个提交保持原始 commit ID，未经过 cherry-pick 或改写：

- `2fa9a44`：为 FP8 转换增加 `transform_ue8m0` 语义透传；与已有 accelerator 设备改动自动合并，无新增 CUDA 设备假设。
- `876cd89`：增加 ROCm INT4 QAT kernel 支持；该原生扩展仍是 CUDA/ROCm 专用边界，没有真实 MUSA kernel，因此不做机械适配。
- `00986d7`：兼容 latest Megatron tokenizer/参数接口；其 `megatron_utils/arguments.py` 新导入路径需要 accelerator 先于 Megatron 执行，import barrier 已折叠进核心提交 `434549f`。
- `41014d1`：新增 debug train/rollout 路径冲突校验，无设备或通信适配需求。

## 2. `434549f`（原 `6a830964`）：核心后端适配

### 2.1 Accelerator 包与 bootstrap

新增：

- `slime/utils/accelerator/musa.py`
- `slime/utils/accelerator/__init__.py`

`musa.py` 负责：

- 检测 `torch.musa`、CUDA 和 CPU；
- 生成设备对象与设备名称，设置/获取当前设备；
- 同步、cache、IPC、stream、显存统计和设备属性；
- 在 `CUDA_VISIBLE_DEVICES` 与 `MUSA_VISIBLE_DEVICES` 间选择并解析物理/局部设备 ID；
- 将默认训练 backend 的 `nccl` 映射为 `mccl`；
- 将 MUSA 权重更新 backend 映射为 `cpu:gloo,musa:mccl`；
- 根据 `MUSA_VISIBLE_DEVICES`、`MUSA_PATCH_PATH` 或已可用的 `torch.musa` 条件化导入 `musa_patch`；
- 用进程内状态保证 `_try_import_musa_patch()` 只成功导入一次。

如果环境变量明确请求 MUSA、但 `torch.musa` 不可用，`process_group_backend()` / `weight_update_backend()` 会抛出错误，避免错误回退到 NCCL。

`__init__.py` 把 `slime.utils.accelerator` 包模块别名到 `musa.py` 的实现模块。这样既保留统一的：

```python
from slime.utils import accelerator
```

也使测试 monkeypatch 直接作用于真实实现模块。

### 2.2 导入顺序

需要在 Megatron/SGLang 之前完成 patch 的独立路径，会先导入 accelerator，并用 `# isort: split` 阻止 formatter 把它移到第三方依赖之后。核心提交中明确设置这类 import barrier 的文件包括：

- `slime/ray/train_actor.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `slime/backends/megatron_utils/__init__.py`
- `slime/backends/megatron_utils/actor.py`
- `slime/backends/megatron_utils/data.py`
- `slime/backends/megatron_utils/hf_checkpoint_saver.py`
- `slime/backends/megatron_utils/server/logprob_utils.py`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`
- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`
- `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `slime_plugins/models/flash_dot_product_attention.py`
- `slime_plugins/models/qwen3_5.py`、`qwen3_5_vl.py`、`qwen3_next.py`
- `tools/convert_hf_to_torch_dist.py`、`tools/convert_to_hf.py`

其中 `megatron_utils/__init__.py` 直接把 accelerator 放在 DeepEP 导入前，其余关键入口主要依靠显式 split。这个约束是“accelerator 的模块代码先执行”，不是简单要求 import 文本出现在文件中；后续修改这些文件时必须复查实际顺序。

### 2.3 设备、内存与 profiler

通用运行路径中的直接 CUDA 调用被替换为 accelerator API：

- Megatron rollout 数据、packed sequence、logprob server 数据移动到当前加速器；
- HF 权重 iterator、distributed/tensor/disk-delta 权重更新使用当前设备、stream 和同步接口；
- checkpoint 写入、tensor 权重传输使用统一 IPC/cache 回收；
- `memory_utils.py`、`profile_utils.py`、`routing_replay.py`、`tensor_backper.py` 使用统一的显存、同步和设备接口；
- profiler 活动按 CUDA/MUSA 选择，memory snapshot 使用当前加速器的 memory 模块。

`434549f` 建立 profiler 的通用选择框架；MUSA OOM observer 私有接口的具体保护由后续临时 patch `f9a662b` 补充。

### 2.4 分布式与可重载进程组

- `process_group_backend("nccl")` 在 MUSA 可用时返回 `mccl`。
- `weight_update_backend("nccl")` 在 MUSA 可用时返回 `cpu:gloo,musa:mccl`。
- `reloadable_process_group.py` 将原先仅面向 NCCL 的判断泛化为 accelerator backend，识别 `nccl`、`mccl` 和复合 MUSA backend。
- WORLD 与子组销毁/重建继续保存实际 store、rank、world size、backend 和 timeout，避免 wake/reload 时把 MUSA 组按 NCCL 重建。

### 2.5 Ray 与 SGLang

- `slime/ray/train_actor.py` 根据 Ray GPU ID 和当前 visible-devices 环境计算局部设备，设置当前 accelerator，并使用映射后的训练 backend。
- `slime/ray/utils.py` 增加 `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES`，并通过统一设备属性获取物理设备标识；没有 UUID 时回退为设备序号。
- `slime/backends/sglang_utils/sglang_engine.py` 通过统一映射计算 engine base GPU ID，不再只按 CUDA 环境解释。

### 2.6 Megatron 训练与模型路径

改动覆盖：

- `actor.py`：wake barrier 设备、rollout token 预搬运；
- `data.py`：`cu_seqlens`、context-parallel 数据和 CPU 回传 tensor；
- `server/logprob_utils.py`：token、loss mask 和 label token tensor；
- `hf_checkpoint_saver.py`：IPC/cache 回收；
- `megatron_to_hf/processors/quantizer_compressed_tensors.py`：临时 tensor 和 shape tensor 设备；
- `flash_dot_product_attention.py`、`qwen3_5.py`、`qwen3_5_vl.py`、`qwen3_next.py`：模型参数/buffer 初始化设备。

这里适配的是框架侧通用设备选择，不代表对应模型依赖的所有 CUDA kernel 已经支持 MUSA。

### 2.7 权重更新

核心提交完成两类修改：

1. 临时 tensor、empty bucket、stream、同步、IPC 和 cache 操作切换到 accelerator。
2. 训练侧和 SGLang engine 使用同一个权重更新 backend；MUSA 下为 `cpu:gloo,musa:mccl`。

`434549f` 保留官方基线默认的异步 broadcast 语义。`p2p-broadcast` 等额外协议虽然测试最初在此提交加入，但当前实际传输分支由 `f9a662b` 实现，见第 4 节。

### 2.8 测试与 CI

核心提交新增/扩展：

- `tests/test_accelerator.py`：visible device、backend、MUSA 请求错误、patch 单次导入和 CUDA 环境隔离；
- `tests/test_reloadable_process_group_world.py`：NCCL/MCCL/复合 backend 识别及进程组生命周期；
- `tests/test_update_weight_transport.py`：默认 broadcast 和 p2p 协议；
- `.github/workflows/pr-test.yml.j2` 及生成后的 `.github/workflows/pr-test.yml`：将 accelerator 和 transport 测试注册为 0-GPU CPU 测试。

## 3. `c00cbfd`（原 `902b59ac`）：CI 自包含修复

该提交不新增 MUSA 运行能力，主要解决 CPU CI 因重量级依赖而跳过或导入失败的问题：

- 删除 `slime/backends/megatron_utils/__init__.py` 中未使用的顶层 `torch` import；
- 为 `test_empty_colocated_weight_bucket.py` 的 fake 模块补齐 accelerator；
- 重写 `test_update_weight_transport.py` 的加载方式，用最小 Megatron/Ray/Slime 模块桩直接加载目标文件；
- 移除 `pytest.importorskip("megatron")`，保证两个协议断言在 CPU CI 中真正执行。

因此测试报告应写成 passed，而不是把依赖缺失导致的 skipped 当作验证完成。

## 4. `f9a662b`（原 `200a84b`）：临时 MUSA 运行 patch

该提交是当前环境的 workaround，不属于已收敛的通用 accelerator 设计。它包含：

- `megatron_utils/arguments.py`：在 `validate_args()` 中追加 `MUSA_PATCH_PATH`、导入 `musa_patch`，并直接调用其 `patch_after_import_torch()`；
- `megatron_utils/loss.py`：MUSA 环境跳过两处 logits 必须为 FP32 的断言；
- `megatron_utils/sglang.py`：MUSA 环境不导入 CUDA/DeepGEMM 相关 FP8 helper；
- `update_weight_from_distributed.py`：通过 `UPDATE_MODE` 增加 `broadcast`、`sync-broadcast`、`gloo-broadcast`、`p2p-broadcast`，并补回 receiver 所需的 names/dtypes/shapes/group/version 元数据；
- `sglang_utils/arguments.py`：兼容 current/legacy SGLang 并行参数都不存在的情况；
- `ray/train_actor.py`：MUSA 环境跳过 NVIDIA NVML NUMA affinity；
- `profile_utils.py`：MUSA 下尝试 `_musa_attach_out_of_memory_observer`，缺失时仅警告；
- `convert_hf_to_torch_dist.py`：MUSA 初始化不传 CUDA 风格 `device_id`，并要求 `--use-cpu-initialization`。

需要特别说明两个风险：

1. 核心 accelerator 的 `_import_musa_patch()` 只导入模块，并有测试确保它不会主动重复执行 `patch_after_import_torch()`；临时参数校验代码却会再次显式调用该函数。该行为是环境补丁，不应写成推荐 bootstrap，也应在上游 patch 生命周期稳定后删除。
2. `accelerator.is_musa_environment()` 在设置 MUSA 环境变量但运行时尚不可用时也为真。因此用它跳过 FP32 断言/DeepGEMM import 是宽松兼容保护，不等价于真实 MUSA 能力检测。

`UPDATE_MODE` 当前语义：

| 值 | 训练侧行为 |
| --- | --- |
| 未设置 / `broadcast` | 每个 tensor 在设备组中异步 broadcast，随后等待 handle |
| `sync-broadcast` | 每个 tensor 同步 broadcast |
| `gloo-broadcast` | 将 tensor 搬到 CPU 后同步 broadcast |
| `p2p-broadcast` | 仅组内 global rank 0 按 tag 顺序发送到 rank 1 |

这些模式必须和当前 SGLang MUSA receiver 的 rank、元数据与收包顺序一起验证，不能只凭训练侧测试认定多机链路正确。

## 5. 末尾 `WIP`：备份提交

该提交用于保存当时可运行环境和相关代码，不应算入核心 MUSA 适配。内容包括：

- 新增/更新 `musa_adapter.md`、`musify.md`、`pr_message.md`；
- `.gitignore` 忽略 `musa_example`；
- 新增 `.rayignore`，控制 Ray runtime_env 的仓库打包内容；
- `slime/ray/rollout.py` 兼容旧版 `sglang-router` 不存在 `disable_health_check` 属性；
- `slime/utils/accelerator/__init__.py` 给实现模块补 `__path__`，兼容 CPU 测试中的 `importlib.reload`；
- 新增 `zero2one` reward 路由、实现和测试。

其中 accelerator reload 兼容与 MUSA 测试有关，但 router、reward、Ray 打包和文档属于环境/业务备份内容。后续拆分正式提交时应逐项决定去留。

## 6. 运行前提与配置

1. 安装匹配版本的 MUSA 驱动、`torch_musa`、MCCL，以及与当前 Megatron/SGLang 版本匹配的 `musa_patch`。
2. 设置 `MUSA_PATCH_PATH` 为包含 `musa_patch` 包的目录。
3. 使用 `MUSA_VISIBLE_DEVICES` 指定设备。
4. Ray 不应覆盖调用方设备映射时，设置 `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1`。
5. Slime 原有 `--distributed-backend nccl` 可以保留；MUSA 运行时由 accelerator 转为 MCCL。
6. HF -> torch-dist 转换在 MUSA 下需要 `--use-cpu-initialization`。
7. 只有当前 SGLang receiver 确实要求时才设置临时 `UPDATE_MODE`；默认仍是 `broadcast`。

建议在进程启动早期打印并核对：

```python
from slime.utils import accelerator

print(accelerator.device_type())
print(accelerator.visible_devices_env_key())
print(accelerator.process_group_backend())
print(accelerator.weight_update_backend())
```

## 7. 已知 CUDA-only 边界

当前适配没有修改 C++/CUDA 源文件、编译器或 kernel launch。以下路径仍含实质 CUDA 假设：

- `slime/backends/megatron_utils/alignment/deepgemm_forward.py`
- `slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py`
- `slime/backends/megatron_utils/alignment/deterministic_route_kernels.py`
- `slime/backends/megatron_utils/kernels/int4_qat/`
- `slime_plugins/models/glm5/ops/sparse_mla.py`
- `slime_plugins/models/qwen_gdn_backend.py` 的 FlashQLA/SM capability 路径
- `tools/convert_hf_to_fp8.py`
- `tools/convert_hf_to_int4_direct.py`
- `tools/convert_k2_thinking_int4_to_bf16.py`
- `tools/fp8_cast_bf16.py`

这些功能只有在存在真实 MUSA kernel、依赖支持和数值/性能测试后才能宣称兼容；目前应绕开、增加 capability guard，或给出明确 unsupported 错误。

## 8. 当前验证结果与未覆盖项

2026-08-17 在当前 checkout 执行：

```bash
PYTHONPATH=. pytest -q \
  tests/test_accelerator.py \
  tests/test_reloadable_process_group_world.py \
  tests/test_update_weight_transport.py \
  tests/test_empty_colocated_weight_bucket.py
```

结果为 `21 passed in 9.88s`。这验证了 CPU 上的设备映射、backend 选择、bootstrap 逻辑、进程组生命周期、权重传输分支和空 bucket 行为。

仍需在真实 MUSA 集群验证：

- `torch_musa` 与 `musa_patch` 的真实导入/重复 patch 生命周期；
- MCCL 单机/多机 collective 和 reloadable process group；
- Ray 分配的物理/局部设备映射；
- Megatron 初始化、forward/backward/optimizer；
- SGLang rollout 和连续多轮权重更新，特别是 `p2p-broadcast`；
- MUSA memory snapshot、OOM observer 和 profiler；
- checkpoint、模型转换及目标模型所需的可选 kernel。

在这些集成项完成前，准确表述应是“核心适配逻辑和 CPU 分支测试已完成”，而不是“所有 Slime 功能已完整支持 MUSA”。
