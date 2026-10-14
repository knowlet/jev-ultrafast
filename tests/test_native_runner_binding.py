"""Experimental native execution requires an explicit immutable workspace identity."""

import hashlib

import pytest

from scripts import local_s1_service as service


@pytest.fixture
def paths(tmp_path, monkeypatch):
    workspace, original = tmp_path / "workspace", tmp_path / "original"
    workspace.mkdir()
    original.mkdir()
    monkeypatch.setattr(service, "WORKSPACE", workspace)
    return workspace, original


def binary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture executable; never executed")
    path.chmod(0o700)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_runner_remains_supported_without_experimental_options(paths):
    workspace, original = paths
    runner, digest = binary(original / "native")
    assert service.verified_native_runner(runner, original) == (runner, digest, "original_s1_repository")


def test_workspace_experiment_requires_exact_hash(paths):
    workspace, original = paths
    runner, digest = binary(workspace / "tools/native-prefix/build/native")
    with pytest.raises(ValueError, match="requires --native-runner-sha256"):
        service.verified_native_runner(runner, original)
    with pytest.raises(ValueError, match="differs from"):
        service.verified_native_runner(runner, original, "0" * 64)
    assert service.verified_native_runner(runner, original, digest, True) == (runner, digest, "workspace_experiment")


def test_workspace_symlink_cannot_admit_an_outside_executable(paths):
    workspace, original = paths
    outside, digest = binary(workspace.parent / "outside/native")
    link = workspace / "tools/native-prefix/link"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="must be inside"):
        service.verified_native_runner(link, original, digest, True)


def test_prefix_flag_cannot_accidentally_change_original_runner(paths):
    workspace, original = paths
    runner, digest = binary(original / "native")
    with pytest.raises(ValueError, match="deliberately selected workspace"):
        service.verified_native_runner(runner, original, digest, True)


@pytest.mark.parametrize("invalid", ["g" * 64, "f" * 63, "F" * 64, 123])
def test_invalid_digest_rejected_before_execution(paths, invalid):
    workspace, original = paths
    runner, _ = binary(workspace / "tools/native-prefix/native")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        service.verified_native_runner(runner, original, invalid)
