"""Bounded, read-only Hub discovery. Download clients keep their own transport."""
from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from urllib.parse import quote, urlparse

import httpx

_EXPAND = ['siblings', 'gguf', 'safetensors', 'gated', 'downloads', 'library_name', 'pipeline_tag', 'tags']


def _info(value):
    value = dict(value)
    value['siblings'] = [SimpleNamespace(rfilename=s.get('rfilename'), size=s.get('size')) for s in value.get('siblings') or []]
    return SimpleNamespace(**value)


class HubDiscovery:
    """One HTTP attempt per page/detail, no redirects/retries; bounded body/deadline.

    Pagination follows only same-origin model-list links. Callers persist the next
    URL so advancing a page never downloads its preceding prefix again.
    """
    def __init__(self, *, endpoint=None, token=None, transport=None):
        from huggingface_hub import constants
        from huggingface_hub.utils._headers import get_token_to_send
        self.endpoint = (endpoint or constants.ENDPOINT).rstrip('/')
        # Older SDKs resolve the cached token before checking this setting.
        # Anonymous discovery must not even read that credential when disabled.
        self.token = '' if token is None and constants.HF_HUB_DISABLE_IMPLICIT_TOKEN else (get_token_to_send(token) or '')
        # SDK user-agent construction can fetch an agent registry. Discovery
        # owns every HTTP attempt, so do not invoke that telemetry path.
        self.headers = {'user-agent': 'abstractcore-hub-discovery'}
        if self.token:
            self.headers['authorization'] = 'Bearer ' + self.token
        self.scope = hashlib.sha256((self.endpoint + '\0' + (self.token or '')).encode()).hexdigest()
        self.transport = transport

    def _get(self, url, *, params=None):
        target, origin = urlparse(url), urlparse(self.endpoint)
        if (target.scheme, target.netloc) != (origin.scheme, origin.netloc) or (target.path != '/api/models' and params is None):
            raise ValueError('Invalid Hub discovery pagination URL')
        headers = self.headers
        deadline = time.monotonic() + 20
        with httpx.Client(transport=self.transport, timeout=httpx.Timeout(5, connect=3), follow_redirects=False, headers=headers) as client:
            with client.stream('GET', url, params=params) as response:
                response.raise_for_status()
                parts, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 16 * 1024 * 1024 or time.monotonic() > deadline:
                        raise TimeoutError('Hub discovery response exceeded its size or time budget')
                    parts.append(chunk)
                return json.loads(b''.join(parts)), response.links.get('next', {}).get('url')

    def list_models_page(self, *, search, filter, limit, cursor=None):
        params = None if cursor else {'search': search, 'filter': filter, 'sort': 'downloads', 'direction': -1, 'limit': limit, 'expand': _EXPAND}
        values, next_url = self._get(cursor or self.endpoint + '/api/models', params=params)
        if not isinstance(values, list):
            raise ValueError('Unexpected Hub search response')
        return [_info(value) for value in values[:limit]], next_url

    def model_info(self, repo_id, *, files_metadata=True):
        value, _ = self._get(self.endpoint + '/api/models/' + quote(repo_id, safe='/'), params={'blobs': 'true'})
        return _info(value)
