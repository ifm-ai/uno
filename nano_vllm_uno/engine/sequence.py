from copy import copy
from enum import Enum, auto
from itertools import count


DECODE_STAT_KEYS = (
    "forwards",
    "accepts",
    "lookaheads",
)


def new_decode_stats() -> dict[str, int]:
    return {key: 0 for key in DECODE_STAT_KEYS}


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        max_completion_tokens: int | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.stats = new_decode_stats()
        self.max_completion_tokens = (
            None
            if max_completion_tokens is None
            else int(max_completion_tokens)
        )

    def to_worker_state(self, include_token_ids: bool):
        """Return only the sequence state a TP model worker needs."""
        return (
            self.seq_id,
            self.status.value,
            self.token_ids if include_token_ids else None,
            self.last_token,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.max_completion_tokens,
        )

    @classmethod
    def from_worker_state(cls, state):
        """Reconstruct a worker-only sequence without advancing its ID counter."""
        (
            seq_id,
            status_value,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens,
            num_cached_tokens,
            block_table,
            max_completion_tokens,
        ) = state
        seq = object.__new__(cls)
        seq.seq_id = seq_id
        seq.status = SequenceStatus(status_value)
        seq.token_ids = token_ids if token_ids is not None else []
        seq.last_token = last_token
        seq.num_tokens = num_tokens
        seq.num_prompt_tokens = num_prompt_tokens
        seq.num_cached_tokens = num_cached_tokens
        seq.block_table = block_table
        seq.max_completion_tokens = max_completion_tokens
        # TP follower copies do not retain accounting between calls. Only the
        # scheduler-owned driver Sequence accumulates generation statistics.
        seq.stats = new_decode_stats()
        return seq

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def rollback_kv_to(self, target_num_cached_tokens: int) -> None:
        """Restore an exact KV frontier without changing token IDs."""
        min_cached = max(0, len(self) - 1)
        if not min_cached <= target_num_cached_tokens <= self.num_cached_tokens:
            raise ValueError(
                "Invalid KV rollback target: "
                f"minimum={min_cached}, target={target_num_cached_tokens}, "
                f"current={self.num_cached_tokens}"
            )
        self.num_cached_tokens = target_num_cached_tokens

    def extend_tokens(self, tokens: list[int]) -> None:
        """Append token IDs."""
        if not tokens:
            return

        self.token_ids.extend(tokens)
        self.last_token = tokens[-1]
        self.num_tokens += len(tokens)
