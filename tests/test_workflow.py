"""The pre-screen sweep workflow's invariants, checked without running the workflow.

The first dispatch of `prescreen-sweep.yml` failed for two reasons that no unit test would have
caught and that no one reads the YAML closely enough to notice: every cell was held to the same
three checkpoints, so the untrained baseline -- the one row a runner without checkpoints can
reproduce -- was blocked by data it never reads; and no cell passed `--output`, so no cell wrote a
table and the merge had nothing to compare. Both are properties of the shell the workflow ships.

Checking them here means reading the workflow, not a copy of it: the assertions are about the text
the runner would execute. Deliberately dependency-free -- PyYAML is not in `requirements.txt`, and
the workflow itself installs nothing before calling `sweep.py` -- so a small indent-based reader
stands in for a YAML parser. It knows just enough about the file's shape (a job's `steps:` list,
each entry starting with `- `) to hand back one step's block of lines.
"""

from __future__ import annotations

import pathlib
import unittest

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1] / ".github/workflows/prescreen-sweep.yml"
)

#: The dev-seed-reproducible row of the reference table needs no checkpoint, and it is the reason
#: the per-cell data plumbing exists at all.
REFERENCE = "docs/prescreen-reference/spike_sum-settle5-dev.json"


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def step_lines(text: str, step: str) -> list[str]:
    """The lines of one `- name: <step>` entry in the workflow's `steps:` lists."""
    lines = text.splitlines()
    start = None
    indent = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("- name:") and stripped.split("- name:", 1)[1].strip() == step:
            start = index
            indent = len(line) - len(stripped)
            break
    if start is None:
        raise AssertionError(f"no step named {step!r} in {WORKFLOW.name}")

    body = []
    for line in lines[start + 1:]:
        stripped = line.lstrip()
        if not stripped:
            body.append(line)
            continue
        if len(line) - len(stripped) <= indent:
            break
        body.append(line)
    return body


def run_block(text: str, step: str) -> str:
    """The `run: |` shell of one step, dedented."""
    body = step_lines(text, step)
    start = next(index for index, line in enumerate(body) if line.strip().startswith("run:"))
    indented = [line for line in body[start + 1:] if line.strip()]
    if not indented:
        raise AssertionError(f"step {step!r} has no run block")
    margin = min(len(line) - len(line.lstrip()) for line in indented)
    return "\n".join(line[margin:] for line in indented)


class PrescreenSweepWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = read_workflow()

    def test_the_cell_writes_the_table_it_is_run_for(self) -> None:
        """A cell that writes no table leaves the merge with nothing, however well it ran."""
        body = run_block(self.text, "Run the cell")
        self.assertIn("--output", body)
        self.assertIn('$CELL_ID.json', body)
        self.assertIn('args+=(--output "runs/prescreen/$CELL_ID.json")', body)

    def test_a_cell_is_held_only_to_the_checkpoints_it_reads(self) -> None:
        """The per-cell list is what lets the untrained baseline run on a runner with no weights."""
        expression = "${{ matrix.checkpoints || inputs.checkpoints }}"
        preflight = "\n".join(step_lines(self.text, "Preflight the data this cell needs"))
        cell = "\n".join(step_lines(self.text, "Run the cell"))
        self.assertIn(f"CHECKPOINTS: {expression}", preflight)
        self.assertIn(f"CHECKPOINTS: {expression}", cell)

    def test_no_checkpoints_is_expressible_despite_the_defaulted_input(self) -> None:
        """`none` is the sentinel: an empty string falls back to the input's default instead."""
        for step in ("Preflight the data this cell needs", "Run the cell"):
            with self.subTest(step=step):
                self.assertIn('"${CHECKPOINTS:-none}" != "none"', "\n".join(step_lines(self.text, step)))

    def test_a_cell_that_ran_and_wrote_no_table_fails(self) -> None:
        """A lost cell should be red, not a green job with a warning nobody reads."""
        upload = "\n".join(step_lines(self.text, "Upload this cell's table"))
        self.assertIn("prescreen-${{ matrix.id }}", upload)
        self.assertIn("if-no-files-found: error", upload)

    def test_the_shard_is_checked_against_the_reference_before_the_run(self) -> None:
        """Rows are matched on the shard's byte hash, and that is worth naming early."""
        step = "Check the shard against the reference before running anything"
        self.assertIn("inputs.reference != ''", "\n".join(step_lines(self.text, step)))
        self.assertIn("shares no shard with the reference", "\n".join(step_lines(self.text, step)))

    def test_the_reference_is_a_real_path_in_this_repository(self) -> None:
        root = WORKFLOW.parents[2]
        self.assertTrue((root / REFERENCE).exists(), f"{REFERENCE} is missing")
        # And the gate compares floats, so an exact reproduction is askable.
        self.assertIn("--tolerance", "\n".join(step_lines(self.text, "Merge the sweep and check it against the reference")))

    def test_cells_come_from_the_dispatch_input(self) -> None:
        self.assertIn("include: ${{ fromJSON(inputs.cells) }}", self.text)
        self.assertIn("cells:", self.text)


if __name__ == "__main__":
    unittest.main()
