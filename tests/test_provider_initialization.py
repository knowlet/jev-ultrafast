"""Local-only startup must not initialize unrelated environment proxy transports."""

import os
import subprocess
import sys
from unittest.mock import Mock, call

from jev_ultrafast import model


def test_local_provider_works_without_optional_socks_dependency():
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith('_proxy')}
    env['ALL_PROXY'] = 'socks5://127.0.0.1:1080'
    # Block socksio explicitly, including if a developer installed it as an extra.
    # MockTransport handles the local request; neither route makes a socket call.
    result = subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['socksio'] = None
import httpx
from jev_ultrafast import model
assert model.CLIENT is None
with httpx.Client(transport=httpx.MockTransport(
    lambda request: httpx.Response(200, json={'local': True})
), trust_env=False) as local:
    model.LOCAL_CLIENT.close()
    model.LOCAL_CLIENT = local
    assert model.post_json('http://127.0.0.1:8767/decide', None, {}) == {'local': True}
assert model.CLIENT is None
try:
    model.remote_client()
except ImportError as error:
    assert 'socksio' in str(error)
else:
    raise AssertionError('Remote proxy configuration must not be silently ignored')
assert model.CLIENT is None
'''], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_remote_client_is_initialized_once_and_reused(monkeypatch):
    response = Mock(status_code=200, is_redirect=False, is_error=False)
    response.json.return_value = {'remote': True}
    client = Mock(post=Mock(return_value=response))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(model, 'CLIENT', None)
    monkeypatch.setattr(model.httpx, 'Client', constructor)
    for _ in range(2):
        assert model.post_json('https://example.invalid/model', 'synthetic-key', {'input': 1}) == {'remote': True}
    assert model.CLIENT is client
    assert client.post.call_args_list == [call(
        'https://example.invalid/model', json={'input': 1},
        headers={'Authorization': 'Bearer synthetic-key'}, follow_redirects=False,
    )] * 2
    constructor.assert_called_once_with(http2=True, timeout=25)
