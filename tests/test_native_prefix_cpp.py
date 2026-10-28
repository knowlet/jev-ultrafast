"""Compiled offline prefix contracts; no model or inference library is executed."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    assert compiler, "A C++17 compiler is required for native prefix regression checks"
    output = tmp_path_factory.mktemp("prefix-cpp") / "test-prefix-plan"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         str(ROOT / "tools/native-prefix/test_prefix_plan.cpp"), "-o", str(output)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return output


def test_compiled_memory_failures_and_chunk_boundaries(planner):
    result = subprocess.run([str(planner)], check=True, capture_output=True, text=True, timeout=30)
    assert "offline prefix memory/geometry checks:" in result.stdout


def test_compiled_planner_replays_all_archived_input_tokens(planner):
    path = ROOT / "benchmark/public/native-prefix-parity-2026-10-14/parity/prepared.jsonl"
    requests = [json.loads(line) for line in path.read_text().splitlines()]
    serialized = "\n".join(
        " ".join(map(str, [len(row["input_ids"]), row["slots"][0], 512, *row["input_ids"]]))
        for row in requests
    ) + "\n"
    result = subprocess.run(
        [str(planner), "--replay"], input=serialized, check=True, capture_output=True, text=True, timeout=30,
    )
    expected = [
        "0 0 601 0:512 512:89",
        "601 512 89 512:89",
        "526 512 79 512:79",
        "591 512 79 512:79",
        "93 0 590 0:512 512:78",
        "590 512 78 512:78",
        "515 512 68 512:68",
        "580 512 68 512:68",
    ]
    assert result.stdout.splitlines() == expected
