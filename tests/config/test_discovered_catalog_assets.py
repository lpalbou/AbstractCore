"""Local models, adapters and incomplete caches are catalogue assets, not omissions."""
import json
import socket

import pytest

from abstractcore.config import model_catalog as mc, model_materializer as mm
from tests.models_engines_fakes import isolate_host, make_hf_repo, synthetic_host


@pytest.fixture
def assets(tmp_path, monkeypatch):
    host = isolate_host(tmp_path, monkeypatch)
    def cfg(**kw): return json.dumps(kw).encode()
    cases = {
        'asset/chat': {'config.json':cfg(architectures=['ExampleForCausalLM'],model_type='example'), 'model.safetensors':b'weights'},
        'asset/sharded': {'config.json':cfg(architectures=['ExampleForCausalLM']), 'model-1.safetensors':b'one', 'model.safetensors.index.json':cfg(weight_map={'a':'model-1.safetensors','b':'model-2.safetensors'})},
        'asset/style': {'adapter_config.json':cfg(task_type='CAUSAL_LM'), 'adapter_model.safetensors':b'adapter'},
        'asset/control': {'config.json':cfg(_class_name='ControlNetModel'), 'diffusion_pytorch_model.safetensors':b'adapter'},
        'asset/encoder': {'config.json':cfg(architectures=['BertModel'],model_type='bert'), 'model.safetensors':b'encoder'},
        'asset/metadata': {'README.md':b'---\npipeline_tag: text-to-image\n---\nmodel card'},
        'asset/opaque': {'weights.ckpt':b'unknown'},
        'asset/speech': {'config.json':cfg(architectures=['ExampleForCausalLM'],model_type='qwen3_tts'), 'model.safetensors':b'speech'},
        'asset/interrupted': {'config.json':cfg(architectures=['ExampleForCausalLM']), 'model.safetensors':b'one'},
        'Qwen/Qwen3-8B': {'config.json':cfg(architectures=['Qwen3ForCausalLM'],model_type='qwen3'), 'model.safetensors':b'qwen'},
    }
    for repo, files in cases.items():
        make_hf_repo(host['hf'], repo, files, incomplete=int(repo=='asset/interrupted'))
    # Discovery/classification never opens an upstream connection.
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k: (_ for _ in ()).throw(AssertionError('unexpected network')))
    return host, cases


def discovered(payload):
    return {r['artifacts'][0]['artifact']:r for r in payload['rows'] if r['source']=='installed'}


def test_all_local_assets_have_first_class_catalogue_rows(assets):
    _, cases = assets
    out=mc.catalog(host=synthetic_host('metal128'))
    rows=discovered(out)
    assert set(rows)==set(cases)-{'Qwen/Qwen3-8B'}
    assert out['counts']['discovered']==len(rows)
    qwen=next(r for r in out['rows'] if r['id']=='qwen3-8b')
    assert sum(a['artifact']=='Qwen/Qwen3-8B' for a in qwen['artifacts'])==1
    for ref,row in rows.items():
        a=row['artifacts'][0]
        assert a['inventory']['present'] is True and a['inventory']['bytes']>0
        assert a['downloadable'] is False and a['recommended'] is False
        assert a['artifact']==ref and a['presence']['location']
    assert rows['asset/style']['asset_kind']=='adapter'
    assert rows['asset/control']['asset_kind']=='adapter'
    assert rows['asset/encoder']['asset_kind']=='component'
    assert rows['asset/opaque']['asset_kind']=='unknown'


def test_incomplete_and_supporting_assets_are_not_runnable_models(assets):
    rows=discovered(mc.catalog(host=synthetic_host('metal128')))
    for ref in ['asset/sharded','asset/interrupted','asset/metadata']:
        a=rows[ref]['artifacts'][0]
        assert a['inventory']['complete'] is False and a['presence']['status']!='installed'
        assert a['fit']['verdict']=='unknown'
    assert rows['asset/sharded']['artifacts'][0]['inventory']['missing_shards']==1
    assert rows['asset/interrupted']['artifacts'][0]['inventory']['incomplete_files']==1
    for ref in ['asset/style','asset/control','asset/encoder','asset/opaque','asset/speech']:
        assert rows[ref]['capabilities']['text'] is False
        assert rows[ref]['artifacts'][0]['fit']['verdict']=='unknown'
    assert rows['asset/chat']['capabilities']['text'] is True
    assert rows['asset/speech']['capabilities']['speech_synthesis'] is True
    assert not rows['asset/speech']['capabilities'].get('thinking')


def test_discovery_obeys_search_engine_tags_and_fit_filters(assets):
    host=synthetic_host('metal128')
    assert set(discovered(mc.catalog('asset/style',host=host)))=={'asset/style'}
    assert set(discovered(mc.catalog(tags=['adapter'],host=host)))=={'asset/style','asset/control'}
    assert not discovered(mc.catalog(engine='ollama',host=host))
    fitted=discovered(mc.catalog(fits=True,host=host))
    assert not set(fitted).intersection({'asset/style','asset/control','asset/metadata','asset/encoder','asset/opaque','asset/interrupted','asset/sharded'})


def test_partial_cache_stays_inspectable_and_deletable(assets):
    host,_=assets
    plan=mm.delete_artifact('huggingface','asset/metadata',dry_run=True)
    assert plan['status']=='planned'
    assert (host['hf']/'models--asset--metadata').exists()


def test_invalid_shard_paths_do_not_probe_outside_snapshot(assets):
    host,_=assets
    make_hf_repo(host['hf'],'asset/bad-index',{'model.safetensors':b'w','model.safetensors.index.json':json.dumps({'weight_map':{'x':'../../outside'}}).encode()})
    row=discovered(mc.catalog(host=synthetic_host('metal128')))['asset/bad-index']
    assert row['artifacts'][0]['inventory']['complete'] is False
