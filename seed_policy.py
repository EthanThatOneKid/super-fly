"""Which seeds may be tuned on, and which may only be reported on.

Every offline number in this experiment used to be measured on whatever seeds the caller
passed, and the default was the same three the closed loop reserves for its published
model-only evaluation. The pre-screen's verdicts were therefore read off the seeds the
eventual claim has to be made on, while the arms themselves were being *chosen* by
watching those same numbers -- a learning-rate sweep, a visual-pathway arm, a margin
calibration method. Re-scoring an arm on the seeds it was selected against does not
un-contaminate it, and a table a reader can re-run is worth little if the arm in it was
picked from the numbers in it.

So the two roles are separate here, and a seed set may never straddle them:

* :data:`DEV_SEEDS` (45-49) -- where iteration happens. Calibration, sweeps, ablations and
  every "has this arm earned an emulator run" verdict are measured here, and may be
  re-measured as often as the work needs, because nothing downstream promises the number
  will stay put.
* :data:`GATE_SEEDS` (42-44) -- the reserved set the closed loop's model-only evaluation is
  reported on. It is **report-only**: fitting a margin, calibrating a decoder or gating a
  candidate on these seeds raises instead of quietly producing a number.

Two failures are errors rather than warnings, because a warning is what the previous
arrangement was:

* a set mixing the two (pass the whole reserved set, or none of it);
* a strict subset of the reserved set, which is neither a claim nor a tuning run.

``replays`` is still the ergonomic entry point for "score this on a few seeds", but it now
walks down :data:`DEV_SEEDS` instead of counting up from a base seed -- counting up from 42
is exactly how a tuning run silently landed on the reserved triple in the first place.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

#: Reserved for the published claim. ``closed_loop_dagger.py`` evaluates model-only on
#: exactly these, so nothing may be fitted, tuned or selected on them.
GATE_SEEDS: Tuple[int, ...] = (42, 43, 44)
#: Where iteration happens. Deliberately longer than the default replay count, so a table
#: that wants more statistical power can spend an unspent dev seed rather than reaching
#: for a reserved one.
DEV_SEEDS: Tuple[int, ...] = (45, 46, 47, 48, 49)

GATE_ROLE = "gate"
DEV_ROLE = "dev"
ROLES = (DEV_ROLE, GATE_ROLE)
#: Replays a table starts with. Three is the smallest count that shows a spread at all,
#: and the fourth and fifth dev seeds are there when three turns out not to be enough.
DEFAULT_REPLAYS = 3


@dataclass(frozen=True)
class SeedSet:
    """One run's evaluation seeds, with the role they are allowed to play."""

    seeds: Tuple[int, ...]
    role: str

    @property
    def report_only(self) -> bool:
        """True when the run may be *reported* but nothing may be *chosen* from it."""
        return self.role == GATE_ROLE

    def to_dict(self) -> Dict[str, object]:
        """The provenance a report carries, including the sets this one was drawn from."""
        return {
            "seeds": list(self.seeds),
            "role": self.role,
            "report_only": self.report_only,
            "dev_seeds": list(DEV_SEEDS),
            "reserved_gate_seeds": list(GATE_SEEDS),
            "note": describe(self),
        }


def describe(seed_set: SeedSet) -> str:
    """One line a reader can act on, printed in every table header."""
    if seed_set.role == GATE_ROLE:
        return (f"reserved gate seeds {list(seed_set.seeds)}: report-only (the emulator "
                "evaluation is where the claim is made)")
    return (f"dev seeds {list(seed_set.seeds)}: tuning only (no claim may be made from "
            "this run; it is not the reserved set)")


def parse_seeds(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated seed list, the form every CLI here accepts."""
    return tuple(int(token.strip()) for token in str(value).split(",") if token.strip())


def dev_seeds(count: int = DEFAULT_REPLAYS) -> Tuple[int, ...]:
    """The first ``count`` dev seeds, refusing to walk off the end of the dev range."""
    if count < 1:
        raise ValueError("at least one dev seed is required")
    if count > len(DEV_SEEDS):
        raise ValueError(
            f"{count} seeds would run past the dev range {list(DEV_SEEDS)}; the next seeds "
            f"up are the reserved gate set {list(GATE_SEEDS)}, which is report-only. Widen "
            "DEV_SEEDS instead of spending a reserved seed on iteration."
        )
    return DEV_SEEDS[:count]


def reserved_seeds(seeds: Iterable[int]) -> Tuple[int, ...]:
    """The reserved seeds inside ``seeds``, in reserved order."""
    present = set(int(seed) for seed in seeds)
    return tuple(seed for seed in GATE_SEEDS if seed in present)


def assert_not_reserved(seeds: Sequence[int], purpose: str) -> None:
    """Refuse to fit anything on the reserved set.

    Calibration is iteration by definition -- it picks a value by looking at the data --
    so a margin fitted on the seeds the claim is reported on would leak the claim back
    into the thing being claimed.
    """
    reserved = reserved_seeds(seeds)
    if reserved:
        raise ValueError(
            f"{purpose} may not use the reserved gate seeds {list(reserved)}: fitting on the "
            f"seeds the claim is reported on leaks the claim. Fit on dev seeds "
            f"{list(DEV_SEEDS)} and report on the reserved set."
        )


def resolve_seeds(seeds: Optional[Sequence[int]] = None, replays: Optional[int] = None,
                  role: Optional[str] = None) -> SeedSet:
    """Resolve a seed set from explicit seeds, a replay count, or nothing at all.

    Defaults to the dev range: a caller who did not say which seeds it wanted is
    iterating, and iteration does not get to touch the reserved set by accident.

    Idempotent: an already-resolved :class:`SeedSet` passes straight through, so a caller
    may hand one to any layer without checking which form that layer wants.
    """
    if isinstance(seeds, SeedSet):
        if replays is not None and int(replays) != len(seeds.seeds):
            raise ValueError(
                f"replays={replays} does not match the {len(seeds.seeds)} seeds in "
                f"{list(seeds.seeds)}"
            )
        if role is not None and role != seeds.role:
            raise ValueError(
                f"role {role!r} does not match the {seeds.role!r} role of "
                f"{list(seeds.seeds)}"
            )
        return seeds

    if seeds is None:
        chosen = dev_seeds(DEFAULT_REPLAYS if replays is None else int(replays))
    else:
        chosen = tuple(int(seed) for seed in seeds)
        if not chosen:
            raise ValueError("an evaluation seed set cannot be empty")
        if len(set(chosen)) != len(chosen):
            raise ValueError(
                f"evaluation seeds repeat ({list(chosen)}); a repeated seed double-counts "
                "one draw and would make the per-replay spread a lie"
            )
        if replays is not None and int(replays) != len(chosen):
            raise ValueError(
                f"replays={replays} does not match the {len(chosen)} seeds given "
                f"({list(chosen)})"
            )
        reserved = reserved_seeds(chosen)
        if reserved and len(reserved) != len(chosen):
            raise ValueError(
                f"evaluation seeds straddle the tuning/claim split: {list(chosen)} mixes the "
                f"reserved gate set {list(GATE_SEEDS)} with dev seeds. Pass the whole "
                "reserved set (a report-only claim) or none of it (a tuning run)."
            )
        if reserved and set(chosen) != set(GATE_SEEDS):
            missing = [seed for seed in GATE_SEEDS if seed not in chosen]
            raise ValueError(
                f"a gate claim is measured on the whole reserved set {list(GATE_SEEDS)}; "
                f"{list(chosen)} is missing {missing}. A subset is neither a claim nor a "
                "tuning run."
            )

    inferred = GATE_ROLE if set(chosen) == set(GATE_SEEDS) else DEV_ROLE
    if role is not None and role != inferred:
        raise ValueError(
            f"role {role!r} does not match the seeds {list(chosen)}, which are a "
            f"{inferred!r} set; the two roles are inferred from the seeds, never asserted "
            "over them"
        )
    return SeedSet(chosen, inferred)


def gate_seed_set() -> SeedSet:
    """The reserved set, as a :class:`SeedSet` (report-only)."""
    return SeedSet(GATE_SEEDS, GATE_ROLE)
