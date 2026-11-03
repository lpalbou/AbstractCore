"""Adapter declarations stay distinct from full-model capabilities and compatibility checks."""
import json
from pathlib import Path

import jsonschema

from abstractcore.config import model_catalog as mc, model_materializer as mm
from tests.models_engines_fakes import FakeHfApi, isolate_host, make_hf_repo, synthetic_host


def test_adapter_metadata_retains_declared_base_usage_and_provenance(tmp_path, monkeypatch):
    host = isolate_host(tmp_path, monkeypatch)
    make_hf_repo(host['hf'], 'org/style', {
        'adapter_config.json': json.dumps({'peft_type':'LORA','task_type':'CAUSAL_LM','base_model_name_or_path':'Org/Base-8B','r':16,'target_modules':['q_proj','v_proj']}).encode(),
        'adapter_model.safetensors': b'weights',
        'README.md': b'---\nbase_model: Other/Base\ntrigger_words:\n- portrait style\n- other trigger\n---\n',
    })
    inv = next(r for r in mm.list_installed()['rows'] if r['artifact']=='org/style')
    a = inv['adapter']
    schema = json.loads((Path(mc.__file__).parents[1]/'assets/adapter_metadata.schema.json').read_text())
    jsonschema.validate(a, schema)
    assert a['type']=='lora' and a['peft_type']=='LORA'
    assert a['base_models']==['Org/Base-8B'] and a['sources']['base_models']=='adapter_config'
    assert a['tasks']==['text_generation'] and a['rank']==16 and a['target_modules']==['q_proj','v_proj']
    assert a['trigger_words']==['portrait style','other trigger']
    row = next(r for r in mc.catalog(host=synthetic_host('metal128'))['rows'] if r['id']=='installed:huggingface:org/style')
    assert row['asset_kind']=='adapter' and row['capabilities']['text'] is False
    assert row['artifacts'][0]['inventory']['adapter']==a


def test_adapter_relation_is_not_a_generic_finetune_or_name_guess(tmp_path, monkeypatch):
    host = isolate_host(tmp_path, monkeypatch)
    for name, relation in [('adapter','adapter'),('full-lora-named-model','finetune')]:
        make_hf_repo(host['hf'], 'org/'+name, {
            'config.json':json.dumps({'architectures':['ExampleForCausalLM']}).encode(),
            'model.safetensors':b'weights',
            'README.md':f'---\nbase_model:\n- Org/Base\nbase_model_relation: {relation}\npipeline_tag: text-generation\n---\n'.encode(),
        })
    rows={r['artifact']:r for r in mm.list_installed()['rows']}
    assert rows['org/adapter']['kind']=='adapter'
    assert rows['org/adapter']['adapter']['base_models']==['Org/Base']
    assert rows['org/full-lora-named-model']['kind']=='model'
    assert 'adapter' not in rows['org/full-lora-named-model']


def test_unknown_adapter_fields_remain_unknown(tmp_path, monkeypatch):
    host = isolate_host(tmp_path, monkeypatch)
    make_hf_repo(host['hf'], 'org/opaque', {'adapter_config.json':b'[]','adapter_model.safetensors':b'weights'})
    a=next(r for r in mm.list_installed()['rows'] if r['artifact']=='org/opaque')['adapter']
    assert a['type']=='unknown' and not a['base_models'] and not a['trigger_words'] and a['rank'] is None


def test_controlnet_is_an_adapter_without_inheriting_chat_capabilities(tmp_path, monkeypatch):
    host = isolate_host(tmp_path, monkeypatch)
    make_hf_repo(host['hf'], 'org/control', {'config.json':b'{"_class_name":"ControlNetModel"}','diffusion_pytorch_model.safetensors':b'w'})
    r=next(r for r in mm.list_installed()['rows'] if r['artifact']=='org/control')
    assert r['kind']=='adapter' and r['adapter']['type']=='controlnet'


def test_explicit_hub_adapter_is_not_mislabelled_as_a_full_model(tmp_path, monkeypatch):
    isolate_host(tmp_path, monkeypatch)
    api=FakeHfApi(repos={'Org/Style':{'files':{'adapter_config.json':100,'adapter_model.safetensors':10000}}})
    row=next(r for r in mc.catalog('Org/Style',hub=True,hf_api=api,hub_cache=tmp_path/'hub.json',host=synthetic_host('cuda24'))['rows'] if r['source']=='hf_search')
    assert row['asset_kind']=='adapter' and row['artifacts'][0]['asset_kind']=='adapter'
    assert row['artifacts'][0]['fit']['verdict']=='unknown'
