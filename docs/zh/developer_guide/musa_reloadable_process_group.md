# MUSA 可重载 Process Group 改动说明

## 背景

`ReloadableProcessGroup` 会保存 `torch.distributed.new_group()` 的创建参数，在训练进程休眠时销毁通信组，并在唤醒时使用原参数重新创建。原实现只按 NCCL 处理设备通信组，因此无法完整覆盖 MUSA 使用的 MCCL backend。

## line 160 附近的改动

改动后的 `new_group()` wrapper 会先读取调用者指定的 backend；没有显式指定时，则继承 WORLD group 的 backend：

```python
explicit_backend = args[2] if len(args) >= 3 else kwargs.get("backend")
backend = str(explicit_backend) if explicit_backend is not None else str(dist.get_backend())
normalized_backend = accelerator.process_group_backend(backend) if backend == "nccl" else backend
```

### `backend` 与 `normalized_backend` 的含义

`backend` 表示调用方原本请求的 backend。如果 `new_group()` 显式传入了 backend，代码从第三个位置参数或 `backend=` 关键字参数中读取；如果没有显式传入，则通过 `dist.get_backend()` 继承默认 WORLD group 的 backend。

`normalized_backend` 表示经过兼容性规范化后，实际应该交给 PyTorch 创建 subgroup 的 backend。只有逻辑默认值 `nccl` 依赖当前 accelerator；Gloo 和已经规范化的 backend 会直接返回，不触发 accelerator 选择。典型结果如下：

| Accelerator | `backend` | `normalized_backend` |
| --- | --- | --- |
| CUDA | `nccl` | `nccl` |
| MUSA | `nccl` | `mccl` |
| MUSA | `mccl` | `mccl` |
| CUDA/MUSA | `gloo` | `gloo` |

当选中的 accelerator 是 MUSA 时，逻辑 backend `nccl` 会被规范化为 `mccl`；CUDA 环境仍保持 `nccl`，显式 `gloo` 和已经规范化的 backend 原则上保持不变。代码同时支持 `new_group()` 的第三个位置参数和 `backend=` 关键字参数。当前 PyTorch 签名中第三个参数确实是 `backend`。

### 为什么只在 backend 发生变化时修改参数

```python
if normalized_backend != backend:
```

这个判断将兼容性改写限制在确实需要转换的调用上。CUDA 下的 `nccl -> nccl`、MUSA 下已经正确指定的 `mccl -> mccl`，以及 `gloo -> gloo` 都不需要重建参数；MUSA 下的 `nccl -> mccl` 才需要进入分支。这样能够保持调用方原来的参数形态，并避免给无需转换的调用额外注入 backend。

### 位置参数和关键字参数的处理

`torch.distributed.new_group()` 的第三个位置参数是 backend。如果调用方使用位置参数：

```python
dist.new_group(ranks, timeout, "nccl", pg_options)
```

代码会只替换索引 2，并保留前后的其他参数：

```python
args = (*args[:2], normalized_backend, *args[3:])
```

在 MUSA 环境中，上述调用等价于：

```python
dist.new_group(ranks, timeout, "mccl", pg_options)
```

如果调用方通过 `backend=` 关键字传入，或者没有显式 backend、但继承到的 WORLD backend 需要转换，则使用：

```python
kwargs = {**kwargs, "backend": normalized_backend}
```

这会创建新的 `kwargs` 字典并设置规范化后的 backend，不直接修改传入的原字典。例如：

```python
{"ranks": [0, 1], "backend": "nccl"}
```

在 MUSA 环境下会变为：

```python
{"ranks": [0, 1], "backend": "mccl"}
```

完成参数改写后还需要执行：

```python
backend = normalized_backend
```

后续的 Gloo 分支判断、日志和 `ReloadableProcessGroup.group_info` 都使用 `backend`。更新该变量可以保证实际创建的 group 与保存的元数据一致，避免出现首次创建使用 MCCL、唤醒重建却根据旧元数据退回 NCCL 的情况。

规范化必须发生在原始 `old_new_group()` 调用之前，否则 MUSA 环境会先尝试用错误的 NCCL backend 创建 subgroup。规范化后的 `args` 或 `kwargs` 还会保存在 `ReloadableProcessGroup.group_info` 中，因此唤醒时通过 `old_new_group()` 重建 subgroup，仍然使用首次创建成功的 MCCL backend，而不会退回 NCCL。

## 同一提交中的配套改动

该提交还将只识别 `nccl` 的 `_uses_nccl()` 改为 accelerator backend 检测，使以下 backend 都能进入 WORLD 销毁和重建流程：

- CUDA/NCCL；
- MUSA/MCCL；
- 包含 MCCL 的复合 backend，例如 `cpu:gloo,musa:mccl`。

相关状态变量和函数名也由 `nccl_world_destroyed`、`_destroy_default_nccl_process_group()` 泛化为 accelerator 语义。这些改名与实际支持范围一致，不改变 WORLD 临时切换到 Gloo、再恢复原设备 backend 的生命周期。

## 合理性结论

改动目的和主要顺序是合理的：backend 映射集中由 accelerator 决定，首次创建和唤醒重建使用同一组已规范化参数，CUDA 路径保持原语义，MUSA 路径能够使用 MCCL。

原始 Slime 对 CPU/Gloo 场景不依赖 accelerator。MUSA 适配最初对所有 backend 都调用 `accelerator.process_group_backend()`，会让纯 CPU 环境中的合法 Gloo subgroup 因 accelerator 不可用而失败。当前实现通过内联条件恢复了原语义：仅逻辑 `nccl` 进入 accelerator 映射，`gloo`、`mccl` 和复合 backend 等无需转换的值直接通过。

对应回归测试验证了 Gloo 路径不会调用 accelerator 选择，同时验证逻辑 `nccl` 在 MUSA 映射下仍能转换为 `mccl`。因此当前逻辑同时保持了原始 CPU-only 行为和 MUSA/CUDA backend 适配。
