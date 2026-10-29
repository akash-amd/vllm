# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP prefill delayer.

The DP ranks of a MoE model step in lockstep, so a step in which any rank runs
prefill is a long step for every rank. Once per engine step all ranks
all-reduce their prefill state and take the same decision: FIRE lets every rank
with prefill work run it in the next step, HOLD defers all prefill (new requests
and in-progress chunks) on every rank. The rules follow ATOM's PrefillDelayer.

Two more of ATOM's rules, off unless their environment variables are set:
- VLLM_DP_PREFILL_DELAYER_MAX_QUEUE_MS: FIRE once the oldest admittable waiting
  request on any rank has waited this long since arrival (after the protected
  steps), as ATOM_PREFILL_DELAYER_MAX_QUEUE_MS does.
- VLLM_DP_PREFILL_BALANCE=1: DP prefill cost balancing. Every rank estimates the
  cost of the prefill it would run next with its whole token budget, the costs
  ride the same all-reduce, and on FIRE a rank whose cost is above
  min(max cost, (1 + slack) x mean cost over the ranks with prefill) caps its
  prefill tokens for that step, never below a floor (ATOM_DP_PREFILL_BALANCE_*).
"""

import os
import time
from collections import Counter
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler

# FIRE once the pending prefill tokens fill this fraction of the prefillable
# ranks' token budgets.
FILL_TARGET = 0.9
# Upper bounds on HOLD, in steps.
TTFT_MAX_HOLD_STEPS = 200
PARTIAL_MAX_HOLD_STEPS = 100
STALL_STEPS = 10
# FIRE when a prefillable rank's KV cache usage reaches this fraction.
KV_HIGH_USAGE = 0.9
# Log the FIRE / HOLD reason counts every this many steps (0 disables).
LOG_EVERY_STEPS = int(os.environ.get("VLLM_DP_PREFILL_DELAYER_LOG_EVERY", "2000"))
_MAX_QUEUE_MS = os.environ.get("VLLM_DP_PREFILL_DELAYER_MAX_QUEUE_MS", "")
MAX_QUEUE_S = float(_MAX_QUEUE_MS) / 1000.0 if _MAX_QUEUE_MS else None
BALANCE = os.environ.get("VLLM_DP_PREFILL_BALANCE", "0") == "1"
# ATOM's chunk cost: k tokens starting at position s cost k * (WEIGHT + s + k / 2),
# a per-token term plus attention over the context before each token.
BALANCE_TOKEN_WEIGHT = int(
    os.environ.get("VLLM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", "360000")
)
BALANCE_SLACK = float(os.environ.get("VLLM_DP_PREFILL_BALANCE_SLACK", "0.1"))
BALANCE_MIN_TOKENS = int(os.environ.get("VLLM_DP_PREFILL_BALANCE_MIN_TOKENS", "512"))
# Token slots a running decode takes from the budget when estimating the prefill
# room (DSpark K6 verifies up to 7 rows per request).
DECODE_ROWS = 7
# Entries of the per-rank state vector.
NUM_STATE = 7


def local_state(scheduler: "Scheduler", token_budget: int) -> list[int]:
    """[prefillable, pending prefill tokens (capped), decodes, KV high,
    chunk in progress, prefill ran since the last call, oldest waiting request
    older than MAX_QUEUE_S] of one rank."""
    pending = decodes = partial = waiting = 0
    oldest_arrival = None
    for request in scheduler.running:
        if request.is_prefill_chunk:
            partial = 1
            pending += max(request.num_tokens - request.num_computed_tokens, 0)
        else:
            decodes += 1
    for request in scheduler.waiting:
        if request in scheduler.deferred_waiting:
            continue
        if not waiting:
            # The queue is in arrival order, so its first admittable request
            # is the oldest.
            oldest_arrival = getattr(request, "arrival_time", None)
        waiting = 1
        if pending >= token_budget:
            break
        pending += max(request.num_tokens - request.num_computed_tokens, 0)
    prefillable = int(bool(partial or waiting))
    kv_high = int(
        bool(prefillable) and scheduler.kv_cache_manager.usage >= KV_HIGH_USAGE
    )
    ran = int(scheduler.ran_prefill)
    scheduler.ran_prefill = False
    queue_hot = 0
    if MAX_QUEUE_S is not None and oldest_arrival is not None:
        age = time.time() - oldest_arrival
        # Ignore clocks that don't look like wall time.
        queue_hot = int(MAX_QUEUE_S <= age < 86400.0)
    return [
        prefillable,
        min(pending, token_budget),
        decodes,
        kv_high,
        partial,
        ran,
        queue_hot,
    ]


def chunk_cost(start: int, tokens: int) -> float:
    return tokens * (BALANCE_TOKEN_WEIGHT + start + tokens / 2)


def prefill_estimate(scheduler: "Scheduler", token_budget: int) -> tuple[int, float]:
    """(tokens, cost) of the prefill this rank would run next with its whole
    budget: chunks in progress first, then the waiting queue, whose cached
    prefixes are looked up without changing any cache state."""
    decodes = sum(1 for r in scheduler.running if not r.is_prefill_chunk)
    room = max(token_budget - DECODE_ROWS * decodes, 0)
    tokens, cost = 0, 0.0
    for request in scheduler.running:
        if not request.is_prefill_chunk:
            continue
        k = min(request.num_tokens - request.num_computed_tokens, room - tokens)
        if k <= 0:
            return tokens, cost
        cost += chunk_cost(request.num_computed_tokens, k)
        tokens += k
    for request in scheduler.waiting:
        if tokens >= room:
            break
        if request in scheduler.deferred_waiting:
            continue
        start = request.num_computed_tokens
        if start == 0:
            try:
                start = scheduler.kv_cache_manager.get_computed_blocks(request)[1]
            except Exception:
                start = 0
        k = min(request.num_tokens - start, room - tokens)
        if k > 0:
            cost += chunk_cost(start, k)
            tokens += k
    return tokens, cost


class PrefillDelayer:
    def __init__(self, dp_size: int, token_budget: int, protected_steps: int):
        self.dp_size = dp_size
        self.token_budget = token_budget
        self.protected_steps = protected_steps
        self._buf = torch.zeros(
            NUM_STATE + (dp_size if BALANCE else 0), dtype=torch.int64
        )
        self._rank: int | None = None
        self._first = True
        self._reasons: Counter[str] = Counter()
        self._steps = 0
        self._capped_steps = 0
        self._cap_ratio_sum = 0.0
        self.reset()
        logger.info(
            "DP prefill delayer: %d ranks, budget %d, %d protected steps, "
            "max queue %s ms, cost balancing %s (weight %d, slack %.2f, floor %d)",
            dp_size,
            token_budget,
            protected_steps,
            _MAX_QUEUE_MS or "off",
            "on" if BALANCE else "off",
            BALANCE_TOKEN_WEIGHT,
            BALANCE_SLACK,
            BALANCE_MIN_TOKENS,
        )

    def reset(self) -> None:
        self._protected_left = 0
        self._hold_steps = 0
        self._stall_steps = 0
        self._prev_pending = -1

    def step(self, scheduler: "Scheduler", dp_group: dist.ProcessGroup) -> bool:
        """Returns whether the next step may run prefill, and sets the
        scheduler's prefill token cap for it. Must run on every DP rank in
        every engine step (a CPU all-reduce on the DP group)."""
        state = local_state(scheduler, self.token_budget)
        own_tokens, own_cost = 0, 0.0
        self._buf.zero_()
        self._buf[:NUM_STATE] = torch.tensor(state, dtype=torch.int64)
        if BALANCE:
            if self._rank is None:
                self._rank = dp_group.rank()
            if state[0]:
                own_tokens, own_cost = prefill_estimate(scheduler, self.token_budget)
            self._buf[NUM_STATE + self._rank] = int(own_cost)
        dist.all_reduce(self._buf, op=dist.ReduceOp.SUM, group=dp_group)
        reduced = self._buf.tolist()
        fire, reason = self.decide(*reduced[:NUM_STATE])
        cap = None
        if fire and BALANCE:
            cap = self.token_cap(reduced[NUM_STATE:], own_tokens, own_cost)
        scheduler.dp_prefill_token_cap = cap
        self._log(fire, reason, dp_group, cap, own_tokens)
        return fire

    @staticmethod
    def token_cap(costs: list[int], own_tokens: int, own_cost: float) -> int | None:
        """This rank's prefill token cap for a FIRE step, or None for no cap."""
        positive = [c for c in costs if c > 0]
        if not positive or own_tokens <= 0 or own_cost <= 0:
            return None
        target = min(max(positive), sum(positive) / len(positive) * (1 + BALANCE_SLACK))
        if own_cost <= target:
            return None
        cap = max(int(own_tokens * target / own_cost), BALANCE_MIN_TOKENS)
        return cap if cap < own_tokens else None

    def _log(
        self,
        fire: bool,
        reason: str,
        dp_group: dist.ProcessGroup,
        cap: int | None,
        own_tokens: int,
    ) -> None:
        if LOG_EVERY_STEPS <= 0:
            return
        self._reasons[("fire_" if fire else "hold_") + reason] += 1
        self._steps += 1
        if cap is not None:
            self._capped_steps += 1
            self._cap_ratio_sum += cap / own_tokens
        if self._steps % LOG_EVERY_STEPS == 0 and dp_group.rank() == 0:
            fires = sum(v for k, v in self._reasons.items() if k.startswith("fire_"))
            logger.info(
                "DP prefill delayer: %d steps, %d fires (%.1f%% of steps held): %s"
                "; rank 0 capped %d fire steps (mean cap %.0f%% of its chunk)",
                self._steps,
                fires,
                100.0 * (self._steps - fires) / self._steps,
                ", ".join(f"{k}={v}" for k, v in sorted(self._reasons.items())),
                self._capped_steps,
                100.0 * self._cap_ratio_sum / max(self._capped_steps, 1),
            )

    def decide(
        self,
        num_prefillable: int,
        pending: int,
        decodes: int,
        kv_high: int,
        partial: int,
        ran: int,
        queue_hot: int = 0,
    ) -> tuple[bool, str]:
        """FIRE (True) or HOLD (False) from the DP-summed state, with the rule
        that decided it."""
        if ran:
            self._protected_left = self.protected_steps
        if self._first:
            self._first = False
            return self._fire("first")
        if self._protected_left > 0:
            self._protected_left -= 1
            if num_prefillable and decodes:
                return False, "protected"
        if num_prefillable == 0:
            return self._fire("no_prefill")
        if decodes == 0:
            return self._fire("no_decode")
        if kv_high:
            return self._fire("kv_high")
        if queue_hot:
            return self._fire("queue_ms")
        if self._hold_steps >= TTFT_MAX_HOLD_STEPS:
            return self._fire("ttft")
        if partial and self._hold_steps >= PARTIAL_MAX_HOLD_STEPS:
            return self._fire("partial")
        if num_prefillable < self.dp_size:
            return self._hold("align")
        if pending >= FILL_TARGET * num_prefillable * self.token_budget:
            return self._fire("fill")
        if pending <= self._prev_pending:
            self._stall_steps += 1
        else:
            self._stall_steps = 0
        self._prev_pending = pending
        if self._stall_steps >= STALL_STEPS:
            return self._fire("stall")
        return self._hold("fill")

    def _fire(self, reason: str) -> tuple[bool, str]:
        self._hold_steps = 0
        self._stall_steps = 0
        self._prev_pending = -1
        return True, reason

    def _hold(self, reason: str) -> tuple[bool, str]:
        self._hold_steps += 1
        return False, reason
