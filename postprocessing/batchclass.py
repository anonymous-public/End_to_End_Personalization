from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Typed Containers
# ---------------------------------------------------------------------------
@dataclass
class BatchResult:
    """Stores the ranking the LLM returned for a batch of ≤ 5 movies."""
    ordered_ids: List[int]  # best ➜ worst
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    scores: dict[int, int]

# @dataclass
# class BatchResult:
#     """Stores the ranking the LLM returned for a batch of ≤ 5 movies."""
#     ordered_ids: List[int]  # best ➜ worst
#     prompt_tokens: int
#     completion_tokens: int
#     total_tokens: int