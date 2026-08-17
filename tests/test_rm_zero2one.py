import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.rm_hub import async_rm
from slime.rollout.rm_hub.zero2one_reward import compute_score, last_boxed_only_string
from slime.utils.types import Sample


NUM_GPUS = 0


@pytest.mark.unit
def test_zero2one_uses_last_balanced_boxed_answer():
    response = r"First try: \boxed{7}. Final answer: \boxed{\frac{8}{2}}"
    assert last_boxed_only_string(response) == r"\boxed{\frac{8}{2}}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "ground_truth", "expected_reward"),
    [
        (r"The answer is \boxed{45}.", 45, 1.0),
        (r"The answer is \boxed{$1,024$}.", "1024", 1.0),
        (r"The answer is \boxed{44}.", 45, 0.0),
        ("The answer is 45.", 45, 0.0),
    ],
)
def test_zero2one_scores_boxed_numeric_answers(response, ground_truth, expected_reward):
    assert compute_score(response, ground_truth)["accuracy_score"] == expected_reward


@pytest.mark.unit
def test_zero2one_dispatch_extracts_parquet_reward_model_label():
    args = SimpleNamespace(custom_rm_path=None, rm_type="zero2one")
    sample = Sample(response=r"The answer is \boxed{45}.", label={"ground_truth": 45, "style": "rule"})

    assert asyncio.run(async_rm(args, sample)) == 1.0


@pytest.mark.unit
def test_zero2one_dispatch_rejects_invalid_mapping_label():
    args = SimpleNamespace(custom_rm_path=None, rm_type="zero2one")
    sample = Sample(response=r"The answer is \boxed{45}.", label={"style": "rule"})

    with pytest.raises(ValueError, match="ground_truth.*answer"):
        asyncio.run(async_rm(args, sample))
