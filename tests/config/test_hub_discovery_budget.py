"""Count underlying discovery HTTP attempts; all responses are local transport fixtures."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from abstractcore.config import model_catalog as mc
from abstractcore.config.hub_discovery import HubDiscovery
from tests.models_engines_fakes import isolate_host, synthetic_host


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, '_hub_cooldowns', {})
    return isolate_host(tmp_path, monkeypatch)


def test_search_is_two_or_one_listing_attempts_with_zero_per_hit_requests(host, tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        assert request.url.path == '/api/models'
        lane = request.url.params['filter']
        return httpx.Response(200, json=[{'id': f'{lane}/Model{i}-GGUF', 'siblings': [{'rfilename': f'Model{i}-Q4_K_M.gguf'}]} for i in range(20)])
    api = HubDiscovery(token='', transport=httpx.MockTransport(respond))
    for kind, expected in [('metal128', 2), ('cuda24', 1), ('cpu16', 1)]:
        cache = tmp_path / (kind + '.json')
        calls.clear()
        kw = dict(hub=True, hf_api=api, hub_cache=cache, host=synthetic_host(kind))
        first = mc.catalog('qwen', **kw)  # many curated matches must not add HTTP work
        assert len(calls) == expected
        assert len([r for r in first['rows'] if r['source'] == 'hf_search']) == expected * 20
        mc.catalog('qwen', extended=True, **kw)
        assert len(calls) == expected
        assert all('blobs' not in request.url.params for request in calls)


def test_page_follows_cursor_once_and_detail_fetches_only_selected_repo(host, tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        if request.url.path != '/api/models':
            return httpx.Response(200, json={'id': 'org/First-GGUF', 'siblings': [{'rfilename': 'First-Q4_K_M.gguf', 'size': 40}, {'rfilename': 'First-Q8_0.gguf', 'size': 80}]})
        second = 'cursor' in request.url.params
        return httpx.Response(200, headers={} if second else {'Link': '<https://huggingface.co/api/models?cursor=next>; rel="next"'},
                              json=[{'id': 'org/Second-GGUF' if second else 'org/First-GGUF', 'siblings': []}])
    api = HubDiscovery(token='', transport=httpx.MockTransport(respond))
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path/'cache.json', host=synthetic_host('cuda24'))
    first = mc.catalog('other', **kw)
    summary = next(r for r in first['rows'] if r['source'] == 'hf_search')
    assert summary['artifacts'][0]['details_required'] and not summary['artifacts'][0]['downloadable']
    second = mc.catalog('other', hub_page=2, **kw)
    assert next(r for r in second['rows'] if r['source'] == 'hf_search')['id'] == 'hf:org/second-gguf'
    assert len(calls) == 2 and dict(calls[-1].url.params) == {'cursor': 'next'}
    detail = mc.catalog('other', detail_repo='org/First-GGUF', extended=True, **kw)
    assert len(calls) == 3 and calls[-1].url.path == '/api/models/org/First-GGUF'
    art = next(r for r in detail['rows'] if r['source'] == 'hf_search')['artifacts'][0]
    assert art['download_bytes'] == 40 and art['downloadable']
    mc.catalog('other', detail_repo='org/First-GGUF', **kw)
    assert len(calls) == 3


def test_retry_after_suppresses_other_queries_and_details_without_retry(host, tmp_path, monkeypatch):
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(mc.time, 'time', lambda: clock[0])
    def respond(request):
        calls.append(request)
        return httpx.Response(429, headers={'Retry-After': '120'}, text='wait')
    api = HubDiscovery(token='', transport=httpx.MockTransport(respond))
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path/'cache.json', host=synthetic_host('metal128'))
    first = mc.catalog('first', **kw)
    assert len(calls) == 1 and first['hub']['retry_at'] == 1120
    for i in range(10):
        for query in ['first', str(i), 'org/Exact']:
            result = mc.catalog(query, **kw)
            assert result['hub']['rate_limited'] and result['hub']['retry_after_seconds'] == 120
        mc.catalog('first', detail_repo='org/Details', **kw)
    assert len(calls) == 1
    clock[0] = 1121
    mc.catalog('after', **kw)
    assert len(calls) == 2


def test_same_query_concurrent_callers_share_cached_listing(host, tmp_path):
    calls = []
    entered, release = threading.Event(), threading.Event()
    def respond(request):
        calls.append(request)
        entered.set()
        assert release.wait(10)
        return httpx.Response(200, json=[])
    api = HubDiscovery(token='', transport=httpx.MockTransport(respond))
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path/'cache.json', host=synthetic_host('cuda24'))
    with ThreadPoolExecutor(4) as pool:
        pending = [pool.submit(mc.catalog, 'same', **kw) for _ in range(4)]
        assert entered.wait(10)
        release.set()
        for future in pending:
            assert future.result()['hub']['ok']
    assert len(calls) == 1


def test_transport_refuses_cross_origin_cursor_without_http():
    calls = []
    api = HubDiscovery(token='secret', transport=httpx.MockTransport(lambda request: calls.append(request)))
    with pytest.raises(ValueError, match='pagination URL'):
        api.list_models_page(search='x', filter='gguf', limit=20, cursor='https://elsewhere.test/api/models')
    assert calls == []


def test_exhausted_mlx_lane_does_not_block_three_gguf_pages(host, tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        if request.url.params.get('filter') == 'mlx':
            return httpx.Response(200, json=[])
        page = int(request.url.params.get('page', 1))
        headers = {'Link': f'<https://huggingface.co/api/models?page={page+1}>; rel="next"'} if page < 3 else {}
        return httpx.Response(200, headers=headers, json=[{'id':f'org/Page{page}-GGUF','siblings':[{'rfilename':f'Page{page}-Q4_K_M.gguf'}]}])
    api = HubDiscovery(token='', transport=httpx.MockTransport(respond))
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path/'cache.json', host=synthetic_host('metal128'))
    for page in (1,2,3):
        result = mc.catalog('page', hub_page=page, **kw)
        assert result['hub']['ok'], result['hub']
        assert result['hub']['search']['has_more'] is (page < 3)
    assert len(calls) == 4  # first two lanes, then one GGUF request per next page


def test_discovery_preserves_disable_implicit_token_policy(monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub.utils import _headers
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_IMPLICIT_TOKEN', True)
    monkeypatch.setattr(_headers, 'get_token', lambda: pytest.fail('implicit token must not be resolved'))
    anonymous = HubDiscovery()
    assert not anonymous.headers.get('authorization')
    explicit = HubDiscovery(token='explicit-token')
    assert explicit.headers['authorization'] == 'Bearer explicit-token'
    assert explicit.scope != anonymous.scope


def test_hub_ratelimit_reset_delay_is_honored_without_retry_after(host, tmp_path, monkeypatch):
    clock = [2000.0]
    monkeypatch.setattr(mc.time, 'time', lambda: clock[0])
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429, headers={'RateLimit':'"api";r=0;t=900'})
    api = HubDiscovery(token='', transport=httpx.MockTransport(respond))
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path/'cache.json', host=synthetic_host('cuda24'))
    assert mc.catalog('first', **kw)['hub']['retry_at'] == 2900
    clock[0] = 2600
    assert mc.catalog('second', **kw)['hub']['retry_after_seconds'] == 300
    assert len(calls) == 1


def test_cli_selected_details_implies_hub_and_forwards_one_repo(host, monkeypatch, capsys):
    import argparse
    from abstractcore.config.models_engines_cli import add_models_subparsers
    calls = []
    def catalog(q=None, **kwargs):
        calls.append((q,kwargs))
        return {'schema':'model_catalog_v1','rows':[]}
    monkeypatch.setattr(mc,'catalog',catalog)
    parser=argparse.ArgumentParser(); add_models_subparsers(parser.add_subparsers())
    args=parser.parse_args(['catalog','--detail-repo','org/Model','--extended','--json'])
    assert args.func(args) == 0
    assert calls == [(None,{'engine':None,'fits':False,'hub':True,'tags':None,'detail_repo':'org/Model','extended':True})]
    assert json.loads(capsys.readouterr().out)['schema'] == 'model_catalog_v1'


@pytest.mark.parametrize('value',['NaN','Infinity','-1','nonsense'])
def test_invalid_quota_delay_has_finite_fallback(host, tmp_path, value):
    api=HubDiscovery(token='',transport=httpx.MockTransport(lambda request:httpx.Response(429,headers={'Retry-After':value})))
    result=mc.catalog('test',hub=True,hf_api=api,hub_cache=tmp_path/'cache.json',host=synthetic_host('cuda24'))
    assert result['hub']['rate_limited'] and 1 <= result['hub']['retry_after_seconds'] <= 60
