"""Explicit policy routing must fail closed rather than fall back to a paid provider."""

from unittest.mock import Mock

import pytest

from jev_ultrafast import model, noul_policy


def test_explicit_noul_policy_dispatches_without_baseline_or_remote(monkeypatch):
    monkeypatch.setenv("S1_LOCAL_POLICY", "noul_v1")
    local = Mock(return_value={"policy_id": "noul_v1"})
    monkeypatch.setattr(noul_policy, "choose_noul", local)
    monkeypatch.setattr(model, "CLIENT", Mock(post=Mock(side_effect=AssertionError("Remote call"))))
    state, history = {"observed": "local"}, []
    assert model.choose(state, "task", history) == {"policy_id": "noul_v1"}
    local.assert_called_once_with(state, "task", history)


def test_unknown_policy_fails_before_any_provider_call(monkeypatch):
    monkeypatch.setenv("S1_LOCAL_POLICY", "unknown")
    post = Mock(side_effect=AssertionError("Provider fallback"))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Unsupported S1_LOCAL_POLICY"):
        model.choose({}, "task", [])
    post.assert_not_called()


def test_explicit_noul_wrapper_preserves_identity_and_request_budget(monkeypatch):
    local = Mock(return_value={"native_calls": []})
    monkeypatch.setattr(noul_policy, "choose_noul", local)
    identity = {"model": "verified-local-package", "temperature": 3.2}
    assert model.choose_noul({}, "task", [], expected_identity=identity, max_native_calls=2) == {"native_calls": []}
    local.assert_called_once_with({}, "task", [], expected_identity=identity, max_native_calls=2)
