"""Registry identity groups exact installed builds without conflating their files."""
import json
from pathlib import Path

import pytest

from abstractcore.architectures.detection import resolve_registered_model_name, get_architecture_format, get_model_capabilities
from abstractcore.config import model_catalog as mc, model_materializer as mm
from tests.models_engines_fakes import isolate_host, synthetic_host

FIXTURE = Path(__file__).parents[1] / 'fixtures' / 'catalog_installed_lmstudio.json'
ROWS = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['artifact'])
def test_reported_installed_variants_have_unambiguous_catalog_identity(row):
    assert mc.catalog_id_for(row['provider'], row['artifact'], metadata=row) == row['expected_catalog_id']


@pytest.mark.parametrize('name', ['qwen3.8-27b-my-finetune', 'qwen3.8-27b-ud', 'qwen3.6-35b-a3b-unlisted-tune', 'lfm3-1.7b-base', 'ternary-bonsai-27b', 'bonsai-2-27b', 'gemma-4-12b-custom'])
def test_unknown_derivatives_do_not_silently_become_the_original(name):
    assert resolve_registered_model_name(name) is None
    assert mc.catalog_id_for('lmstudio', name) is None


def test_every_exact_installed_variant_is_reachable_with_real_size_and_precision(tmp_path, monkeypatch):
    isolate_host(tmp_path, monkeypatch)
    monkeypatch.setattr(mm, 'list_installed', lambda *a, **kw: {'rows': ROWS, 'errors': {}})
    out = mc.catalog(host=synthetic_host('metal128'))
    rows = {r['id']:r for r in out['rows']}
    for local in ROWS:
        row = rows[local['expected_catalog_id']]
        found = [a for a in row['artifacts'] if a['provider'] == local['provider'] and a['artifact'] == local['artifact']]
        assert len(found) == 1, local['artifact']
        art = found[0]
        assert art['presence']['status'] == 'installed'
        assert art['presence']['location'] == local['location']
        assert art['download_bytes'] == local['size_bytes'] and art['size_source'] == 'engine'
        assert art['quant'] == local['quant']
    embed = rows['qwen3-embedding-0.6b']
    assert embed['capabilities']['embedding'] is True and embed['capabilities']['text'] is False
    assert {'qwen3-embedding-0.6b@8bit', 'qwen3-embedding-0.6b@mxfp8'} <= {a['artifact'] for a in embed['artifacts']}
    assert rows['qwen3.6-35b-a3b-hauhaucs-aggressive']['id'] != rows['qwen3.6-35b-a3b']['id']


def test_explicit_precisions_are_not_interchangeable_files():
    assert not mm._matches_installed_id('model@8bit', 'model@4bit')
    assert not mm._lmstudio_row_matches({'modelKey':'model','selectedVariant':'model@8bit','quantization':{'name':'8bit'}}, 'model@4bit')
    assert mm._matches_installed_id('model', 'model@4bit')
    assert mm._matches_installed_id('model@4bit', 'model')


def test_hf_cache_inventory_still_requires_the_checkpoint_probe(tmp_path, monkeypatch):
    isolate_host(tmp_path, monkeypatch)
    calls = []
    def probe(provider, artifact):
        calls.append((provider, artifact))
        return mm.ModelPresence(provider=provider, artifact=artifact, status=mm.PRESENCE_UNKNOWN)
    monkeypatch.setattr(mm, 'probe', probe)
    row = next(r for r in mc.load_seed()['rows'] if r['id'] == 'all-minilm-l6-v2')
    spec = next(a for a in row['artifacts'] if a['provider'] == 'huggingface')
    art = mc._build_artifact(row, spec, {'text':False,'embedding':True}, synthetic_host('metal128'), {}, None, {}, installed={'location':'partial/cache'})
    assert art['presence']['status'] == mm.PRESENCE_UNKNOWN
    assert calls == [(spec['provider'], spec['artifact'])]


def test_new_model_metadata_and_wire_formats_are_specific():
    spark = get_model_capabilities('spark-x2.6-4b')
    assert spark['max_tokens'] == 1048576 and spark['vision_support'] is False and spark['thinking_support'] is True
    fmt = get_architecture_format(spark['architecture'])
    assert fmt['tool_format'] == 'glm_xml' and '<|Bot|>' in fmt['assistant_prefix']
    liquid = get_model_capabilities('LiquidAI/LFM3-1.7B')
    assert liquid['max_tokens'] == 262144 and liquid['vision_support'] is False
    assert get_architecture_format(liquid['architecture'])['tool_calling_format'] == 'qwen3_coder'
    mini = get_model_capabilities('all-minilm-l6-v2@8bit')
    assert mini['model_type'] == 'embedding' and mini['max_output_tokens'] == 0 and mini['embedding_dimension'] == 384
    assert 'speculation' not in get_model_capabilities('bonsai-27b')


def test_new_architecture_tool_payloads_use_existing_parsers():
    from abstractcore.tools.parser import parse_tool_calls
    spark = parse_tool_calls('<tool_call>lookup<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>', model_name='spark-x2.6-4b')
    assert len(spark) == 1 and spark[0].name == 'lookup' and spark[0].arguments['city'] == 'Paris'
    liquid = parse_tool_calls('<tool_call>\n<function=lookup>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>', model_name='lfm3-1.7b')
    assert len(liquid) == 1 and liquid[0].name == 'lookup' and liquid[0].arguments['city'].strip() == 'Paris'
