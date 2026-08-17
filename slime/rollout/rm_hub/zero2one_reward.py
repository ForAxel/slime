import re
from typing import Any


def last_boxed_only_string(text: str) -> str | None:
    index = text.rfind("\\boxed{")
    if index < 0:
        return None

    open_braces = 0
    for position in range(index, len(text)):
        if text[position] == "{":
            open_braces += 1
        elif text[position] == "}":
            open_braces -= 1
            if open_braces == 0:
                return text[index : position + 1]
    return None


def remove_boxed(text: str) -> str:
    prefix = "\\boxed{"
    if not text.startswith(prefix) or not text.endswith("}"):
        raise ValueError(f"Invalid boxed answer: {text}")
    return text[len(prefix) : -1]


def _normalize_numeric_answer(answer: Any) -> str:
    answer = str(answer).strip()
    answer = answer.replace(",", "")
    answer = re.sub(r"^\$|\$$", "", answer)
    return answer.strip()


def is_correct_int(solution_str: str, ground_truth: Any) -> tuple[bool, str]:
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is None:
        return False, "[INVALID]"

    prediction = _normalize_numeric_answer(remove_boxed(boxed_answer))
    try:
        ground_truth = _normalize_numeric_answer(ground_truth)
        return float(prediction) == float(ground_truth), prediction
    except (TypeError, ValueError):
        return False, prediction


def format_reward(solution_str: str) -> float:
    return 0.0


def compute_score(
    solution_str: str,
    ground_truth: Any,
    data_source: str | None = None,
    extra_info: Any = None,
    prompt_str: str | None = None,
    format_score: float = 0.0,
    score: float = 1.0,
    is_train_data: bool = True,
    step: int | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, extra_info, prompt_str, step, kwargs

    solution_str = solution_str.strip()
    is_correct, _ = is_correct_int(solution_str, ground_truth)

    if is_train_data:
        format_score = format_reward(solution_str)
    else:
        format_score = 0.0

    accuracy_score = score if is_correct else 0.0
    total_score = accuracy_score + format_score if is_train_data else accuracy_score
    return {
        "score": total_score,
        "format_score": format_score,
        "accuracy_score": accuracy_score,
    }
