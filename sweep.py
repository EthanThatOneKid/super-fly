"""Merge several pre-screen tables into one comparison, and check them against a reference.

A sweep is a set of runs that differ in exactly one thing -- the statistic
(``--evidence-rule``), the window (``--settle-steps``), or the seed set -- and the job of this
module is to put their rows side by side and say whether the numbers a claim was published on
have moved. It exists because that comparison was being done by hand, once per sweep, from
``runs/prescreen/*.json``: the same arm appears in every table, the statistic and the protocol
are buried in the protocol block, and a table that failed to produce a file simply vanished
from the summary.

Three things make two runs comparable, and all three are in the table rather than in the
command line that produced it:

* the data -- the teacher shard's SHA-256, not its path, so a run in a container and a run on a
  CI runner are the same data or are visibly not;
* the protocol -- stride, settle steps, replay seeds and the seed *role* (dev seeds are the
  tuning set, the reserved seeds are the claim, and a number read off the wrong one is the
  defect the role exists to catch);
* the statistic each row was measured on, which is a forced ``--evidence-rule`` when there is
  one and the per-arm rule the checkpoint carries otherwise.

So rows are keyed by ``(data, protocol, statistic, arm)``. A reference table is diffed on the
numbers a reader would quote -- sequential recall, budget-view recall, jump rate, spurious
jumps, pessimistic ``best_x``, completion -- and the diff is reported with the delta, not just
a pass/fail, because "the number moved by 0.004" and "the number moved by 0.30" are different
findings about whether a cloud runner reproduces a local one.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The numbers a reader would quote from a table, and the tolerance class each is compared
#: under. Recalls and rates are floats: a host that changes floating-point reduction order can
#: move them, which is exactly what a cross-host drift check is for. Completion is a count of
#: replays and is compared exactly.
FLOAT_METRICS = ("jump_sequence_recall", "budget_jump_recall", "jump_rate", "spurious_jumps",
                 "offline_best_x")
EXACT_METRICS = ("offline_completion_rate",)
METRICS = FLOAT_METRICS + EXACT_METRICS


def load_table(path: str) -> Dict[str, object]:
    """Read one pre-screen table, refusing anything that is not one.

    Loudly, and with the path: a sweep whose summary quietly omits a run reads as a sweep that
    ran it, and a truncated artifact is the way that happens.
    """
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or "table" not in report or "protocol" not in report:
        raise ValueError(f"{path} is not a pre-screen table: no 'table'/'protocol' block")
    return report


def shard_checksums(protocol: Dict[str, object]) -> Tuple[str, ...]:
    """The SHA-256 of every shard the run was measured on, as the identity of the data."""
    provenance = protocol.get("dataset_provenance") or {}
    shards = provenance.get("shards") or []
    checksums = tuple(str(shard.get("sha256")) for shard in shards if isinstance(shard, dict))
    if not checksums:
        # A table that cannot name its data cannot be compared with one that can, so this is an
        # error rather than an empty key that would collide with every other unknown dataset.
        raise ValueError("table has no dataset_provenance shard checksums to key its data on")
    return checksums


def forced_statistic(protocol: Dict[str, object]) -> Optional[str]:
    """The statistic forced on every arm, or ``None`` when each arm ran its own."""
    override = protocol.get("evidence_override")
    if not override:
        return None
    return f"{override.get('evidence_rule')}@{override.get('evidence_decay')}"


def cell_key(protocol: Dict[str, object]) -> Tuple[object, ...]:
    """What makes two runs the same cell: the data, the protocol, and the seed role."""
    return (
        shard_checksums(protocol),
        int(protocol.get("stride")),
        int(protocol.get("settle_steps")),
        tuple(int(seed) for seed in (protocol.get("replay_seeds") or ())),
        str(protocol.get("seed_role")),
        forced_statistic(protocol) or "per-arm",
    )


def row_key(entry: Dict[str, object]) -> Tuple[object, ...]:
    """What makes two *rows* the same measurement: the cell, the arm, and its statistic.

    The arm set is deliberately not part of the key. A run that exercises only the untrained
    baseline is comparable with the untrained row of a run that also carried three checkpoints
    -- which is what makes a partial run usable as a reproduce check against a fuller
    published table -- while two runs whose arm of the same name was measured on different
    statistics stay apart, because the statistic is the key.
    """
    return (entry["cell_key"], entry["arm"], entry["statistic"])


def cell_label(protocol: Dict[str, object]) -> str:
    """A short human name for a cell, for the rendered table's first column."""
    statistic = forced_statistic(protocol)
    statistic = statistic if statistic else "per-arm"
    return (f"settle {protocol.get('settle_steps')} / stride {protocol.get('stride')} / "
            f"{statistic}")


def statistic_of(protocol: Dict[str, object], row: Dict[str, object]) -> str:
    """The statistic *this row* was measured on, forced or carried by the arm.

    A table from before the statistic was recorded says so, rather than being labelled with the
    default: it is a different measurement, and one that cannot be compared like for like.
    """
    forced = forced_statistic(protocol)
    if forced:
        return forced
    evidence = row.get("evidence") or {}
    rule = evidence.get("evidence_rule")
    if not rule:
        return "unspecified"
    return f"{rule}@{evidence.get('evidence_decay')}"


def _value(row: Dict[str, object], metric: str) -> object:
    """Read one comparison metric out of a pre-screen row."""
    if metric == "budget_jump_recall":
        budget = row.get("budget") or {}
        recall = (budget.get("jump_recall") or {}).get("mean")
        return None if recall is None else float(recall)
    if metric in ("jump_sequence_recall", "jump_rate", "spurious_jumps", "offline_best_x"):
        spread = row.get(metric) or {}
        mean = spread.get("mean")
        return None if mean is None else float(mean)
    value = row.get(metric)
    return None if value is None else float(value)


def flatten(tables: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """One flat row per ``(cell, arm)``, carrying the numbers and the provenance behind them."""
    rows: List[Dict[str, object]] = []
    for report in tables:
        protocol = report["protocol"]
        key = cell_key(protocol)
        label = cell_label(protocol)
        paired = {entry.get("arm"): entry for entry in (report.get("paired") or [])}
        verdicts = {entry.get("arm"): entry for entry in (report.get("verdicts") or [])}
        for row in report["table"]:
            arm = str(row.get("arm"))
            pairing = paired.get(arm) or {}
            verdict = verdicts.get(arm) or {}
            entry = {
                "cell": label,
                "cell_key": key,
                "statistic": statistic_of(protocol, row),
                "arm": arm,
                "role": row.get("role", "candidate"),
                "seed_role": protocol.get("seed_role"),
                "settle_steps": protocol.get("settle_steps"),
                "stride": protocol.get("stride"),
                "replays": len(protocol.get("replay_seeds") or ()),
                "paired_delta": pairing.get("mean_delta"),
                "paired_improved": pairing.get("improved"),
                "paired_replays": pairing.get("paired_replays"),
                "verdict": bool(verdict.get("pass")),
            }
            for metric in METRICS:
                entry[metric] = _value(row, metric)
            rows.append(entry)
    rows.sort(key=lambda entry: (str(entry["cell"]), entry["role"] != "untrained_baseline",
                                 str(entry["arm"])))
    return rows


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(rows: Sequence[Dict[str, object]]) -> str:
    """The sweep as a markdown table, one row per arm per cell."""
    header = ("cell", "statistic", "arm", "seq recall", "budget recall", "jump rate",
              "spurious", "best_x", "completion", "paired delta", "verdict")
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for entry in rows:
        paired = _number(entry["paired_delta"])
        if paired != "-" and entry["paired_replays"]:
            paired = f"{paired} ({entry['paired_improved']}/{entry['paired_replays']})"
        cells = (
            entry["cell"], entry["statistic"], entry["arm"],
            _number(entry["jump_sequence_recall"]),
            _number(entry["budget_jump_recall"]),
            _number(entry["jump_rate"]),
            _number(entry["spurious_jumps"], 1),
            _number(entry["offline_best_x"], 0),
            _number(entry["offline_completion_rate"], 2),
            paired,
            "pass" if entry["verdict"] else "fail",
        )
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines)


def index(rows: Iterable[Dict[str, object]]) -> Dict[Tuple[object, ...], Dict[str, object]]:
    """Key the rows the way a diff addresses them: ``(cell, arm, statistic)``."""
    return {row_key(entry): entry for entry in rows}


def compare(rows: Sequence[Dict[str, object]], reference_rows: Sequence[Dict[str, object]],
            tolerance: float = 0.01) -> Dict[str, object]:
    """Diff a sweep against a reference sweep, over every metric a reader would quote.

    Nothing is dropped: a cell in one side and not the other is reported as ``missing`` or
    ``extra``, because a sweep that no longer contains the run a claim came from is a finding
    about the sweep, not a reason to compare the rows that happen to line up.
    """
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")
    current, reference = index(rows), index(reference_rows)
    compared = sum(1 for key in current if key in reference)
    moved: List[Dict[str, object]] = []
    for key, entry in sorted(current.items(), key=lambda item: str(item[0])):
        previous = reference.get(key)
        if previous is None:
            continue
        deltas = {}
        for metric in METRICS:
            before, after = previous.get(metric), entry.get(metric)
            if before is None and after is None:
                continue
            if before is None or after is None:
                deltas[metric] = {"reference": before, "current": after,
                                  "delta": None, "moved": True, "reason": "one side is None"}
                continue
            delta = float(after) - float(before)
            limit = tolerance if metric in FLOAT_METRICS else 0.0
            deltas[metric] = {
                "reference": before, "current": after, "delta": delta,
                "moved": abs(delta) > limit,
                "reason": None,
            }
        if any(detail["moved"] for detail in deltas.values()):
            moved.append({"cell": entry["cell"], "arm": entry["arm"], "metrics": deltas})
    # A reference that shares no row with this sweep has not been checked, and calling that a
    # pass would be the silent pass every other guard here exists to prevent. The usual cause is
    # that the data is not the same data -- the shard checksum is in the key -- so both sides'
    # checksums are reported with it.
    unaddressed = compared == 0 and bool(reference)
    return {
        "metrics": list(METRICS),
        "tolerance": tolerance,
        "compared": compared,
        "moved": moved,
        "missing": [{"cell": reference[key].get("cell"), "arm": reference[key].get("arm"),
                     "statistic": reference[key].get("statistic")}
                    for key in sorted(reference, key=str) if key not in current],
        "extra": [{"cell": entry["cell"], "arm": entry["arm"],
                   "statistic": entry["statistic"]}
                  for key, entry in sorted(current.items(), key=lambda item: str(item[0]))
                  if key not in reference],
        "unaddressed": unaddressed,
        "reason": ("nothing compared: this sweep and the reference share no row, which usually "
                   "means the data differs -- check the shard checksums below -- or that the "
                   "protocol, the seeds or the statistic differ"
                   if unaddressed else None),
        "reference_data": sorted({str(key[0]) for key in reference}) if unaddressed else None,
        "current_data": sorted({str(key[0]) for key in current}) if unaddressed else None,
        "ok": not moved and not unaddressed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge pre-screen tables into one comparison and check it for drift"
    )
    parser.add_argument("--tables", nargs="+", required=True,
                        help="Pre-screen table JSONs, one per sweep cell")
    parser.add_argument("--reference", default=None,
                        help="Table JSON (or a directory of them) to diff the sweep against")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="How far a float metric may move before it counts as drift "
                             "(0.0 demands an exact reproduction)")
    parser.add_argument("--output", default=None, help="Write the full report as JSON here")
    args = parser.parse_args(argv)

    rows = flatten([load_table(path) for path in args.tables])

    reference_rows: List[Dict[str, object]] = []
    if args.reference:
        paths = [os.path.join(args.reference, name) for name in sorted(os.listdir(args.reference))
                 if name.endswith(".json")] if os.path.isdir(args.reference) else [args.reference]
        reference_rows = flatten([load_table(path) for path in paths])

    print(render(rows))
    if not args.reference:
        return 0

    # Printed before the exit code, so a caller reading a failed job's log sees which number
    # moved rather than only that one did.
    drift = compare(rows, reference_rows, args.tolerance)
    print()
    print(f"drift vs {args.reference}: {drift['compared']} row(s) compared, "
          f"{len(drift['moved'])} moved beyond {args.tolerance}")
    for entry in drift["moved"]:
        for metric, detail in entry["metrics"].items():
            if detail["moved"]:
                print(f"  {entry['arm']} @ {entry['cell']}: {metric} "
                      f"{detail['reference']} -> {detail['current']} ({detail['delta']})")
    for entry in drift["missing"]:
        print(f"  in the reference, not in this sweep: {entry['arm']} "
              f"({entry['statistic']}) @ {entry['cell']}")
    if drift["reason"]:
        print(f"  {drift['reason']}")
        for side in ("reference_data", "current_data"):
            for checksum in drift[side] or []:
                print(f"    {side.split('_')[0]} shards: {checksum}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "drift": drift}, handle, indent=2)
        print(f"\nwrote {os.path.abspath(args.output)}")
    return 0 if drift["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
