# Introduce runtime-selectable accelerator backends with MUSA support

## Summary

This PR introduces a backend-neutral accelerator layer for Slime and adds a MUSA implementation while preserving the existing CUDA/ROCm execution path.

The PR contains two commits:

- `f63afa1e44e89422033d620c37e92b7a24651b5a feat: introduce extensible accelerator backends with MUSA support`
- `77207515917aeae51282659d187b20a33bdf0a92 test: cover accelerator selection and MUSA weight updates`

The main goal is to centralize device, memory, visibility, and communication-backend differences instead of spreading MUSA-specific branches across training, rollout, profiling, model, and conversion code.

## Motivation

Several shared Slime paths directly assume `torch.cuda`, `CUDA_VISIBLE_DEVICES`, and NCCL. MUSA exposes equivalent device operations through `torch.musa`, uses `MUSA_VISIBLE_DEVICES`, and uses MCCL for accelerator communication.

Without a backend boundary, each shared runtime path would need vendor-specific conditionals and import-order handling.

## What Changed

### Accelerator abstraction and selection

Add `slime/utils/accelerator/` with:

- `Accelerator`, the backend contract used by Slime;
- `TorchAccelerator`, shared delegation for CUDA-like PyTorch namespaces;
- `CUDAAccelerator`, which delegates to the existing `torch.cuda` APIs and keeps NCCL;
- `MUSAAccelerator`, which delegates to `torch.musa` and maps accelerator communication to MCCL.

Backend selection supports:

- `SLIME_ACCELERATOR=<backend>` as an explicit override;
- `MUSA_VISIBLE_DEVICES` or `MUSA_PATCH_PATH` as an explicit MUSA request;
- deterministic availability-based selection when no backend is requested;
- extension and test injection through `register_accelerator()` and `set_accelerator()`.

An unavailable explicitly requested backend fails with a clear error. CPU-only imports remain possible because `initialize_accelerator()` returns `None` when no accelerator is requested or available.

### MUSA bootstrap ordering

Importing `slime.utils.accelerator` alone does not load `musa_patch`. The patch is loaded only after MUSA is selected, but before the MUSA backend is validated and constructed.

The Megatron and SGLang backend packages finalize accelerator selection before importing their third-party runtime modules. An explicit CUDA selection therefore does not import `musa_patch`, even when MUSA-related environment variables are present.

### Runtime integration

Shared runtime paths now use the selected accelerator for the operations they already perform, including:

- device selection and tensor placement;
- synchronization, streams, cache cleanup, IPC collection, and memory statistics;
- Ray visible-device mapping, including numeric IDs and CUDA UUIDs;
- profiler activity and memory-snapshot namespace selection;
- distributed process-group backend selection;
- Megatron/SGLang weight-update group creation;
- affected Qwen model helpers and conversion tools;
- compatibility with older `sglang-router` releases that do not expose `disable_health_check`;
- CPU-only guards around DeepEP synchronization and checkpoint-shard cache cleanup.

This PR does not claim portable MUSA implementations for existing CUDA-only kernels or extensions.

### Distributed backend handling

Logical accelerator backends are mapped as follows:

- CUDA/ROCm keeps `nccl`;
- MUSA maps logical `nccl` to `mccl`;
- MUSA weight-update groups use `cpu:gloo,musa:mccl` so CPU metadata and MUSA tensors can use their respective transports.

This PR changes weight-update group creation to use the selected backend while keeping the existing asynchronous `dist.broadcast` tensor-transfer protocol.

Reloadable process-group handling recognizes registered accelerator communication backends. The `torch.distributed.new_group` wrapper normalizes a logical `nccl` backend, whether supplied explicitly or inherited from the current default; Gloo, already-normalized MCCL, and composite backend strings remain unchanged.

The WORLD teardown/reload lifecycle is generalized from NCCL-specific naming to accelerator process groups while retaining the temporary Gloo WORLD used during communicator teardown. Existing CPU-only and pre-registration behavior is preserved.

### Allocator configuration

Training actors can opt in to expandable allocator segments for the selected PyTorch accelerator:

```bash
SLIME_ENABLE_EXPANDABLE_SEGMENTS=1
```

The value must be `0` or `1` and defaults to `0`, so the CUDA allocator behavior is unchanged unless this option is explicitly enabled.

## CUDA Compatibility

When CUDA is selected:

- device, stream, event, RNG, synchronization, and memory calls delegate to `torch.cuda`;
- logical `nccl` remains `nccl`;
- `CUDA_VISIBLE_DEVICES` remains the visibility source;
- numeric visible-device IDs and CUDA UUIDs map to local device ordinals;
- `musa_patch` is not imported;
- the expandable-segments setting remains disabled by default;
- existing CUDA-only INT4 direct-conversion and Triton FP8-casting paths retain their CUDA behavior.

MUSA-related environment variables intentionally request MUSA when no explicit backend overrides them. A CUDA deployment that exports those variables can select CUDA explicitly with `SLIME_ACCELERATOR=cuda`.

## Tests

The new and updated tests are CPU-runnable. MUSA and MCCL behavior is checked with monkeypatches, a fake `torch.musa` namespace, and backend strings; no test imports `torch_musa`, initializes MCCL, or requires accelerator hardware.

`tests/test_accelerator.py` is registered in the CPU unit-test matrix in both the workflow template and generated workflow.

Validated at the PR tip, commit `77207515917aeae51282659d187b20a33bdf0a92`:

```bash
PYTHONPATH=. pytest -q \
  tests/test_accelerator.py \
  tests/test_empty_colocated_weight_bucket.py \
  tests/test_reloadable_process_group_world.py
```

Result:

```text
20 passed
```

The CPU-only HF checkpoint-saver tests also pass (`6 passed`) without accelerator environment variables.

These tests cover backend selection, strict MUSA bootstrap timing, CUDA delegation, visible-device mapping, NCCL-to-MCCL normalization, CPU/Gloo process-group reload, and empty colocated weight buckets. They do not replace an end-to-end MUSA/MCCL integration run.

`compileall` and `git diff --check` also pass for the PR tip.

## Scope and Validation Limits

- This PR does not add MUSA kernels, compiler support, or vendor runtime packages.
- This PR does not make existing vendor-specific optimized kernels portable; only the shared operations described above are routed through the accelerator abstraction.
- `musa_patch` remains an optional external dependency for MUSA environments that require it and can be located through `MUSA_PATCH_PATH`.
- The focused tests validate selection and backend-routing logic without MUSA/MCCL hardware.
- The validation listed above exercises CPU-only paths; real CUDA/NCCL and MUSA/MCCL end-to-end runs remain separate integration validation.
