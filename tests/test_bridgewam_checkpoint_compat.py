"""Real CPU serialization, import, topology and name migration regressions."""
import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bridgewam.models.wan22.checkpoint_compat import normalize_checkpoint_payload
from tests.test_lbq_spectral_regularization import make_model

ROOT = Path(__file__).resolve().parents[1]


def saved(model, path):
    model.save_checkpoint(path, step=23)
    return torch.load(path, weights_only=True)


def assert_weights(a, b):
    assert a.state_dict().keys() == b.state_dict().keys()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('wrapper', ['native', 'fastwam', 'boe', 'bridgewam', 'state_dict', 'model_state_dict', 'model'])
def test_real_checkpoint_load_keeps_all_weights_buffers_and_metadata(tmp_path, wrapper):
    torch.manual_seed(81)
    original = make_model()
    original.bridge.action_video_kv_layer_mask[0] = False
    original.proprio_encoder.bias.data.fill_(0.123)
    path = tmp_path / 'old_fastwam_boe_step_000023.pt'
    payload = saved(original, path)
    if wrapper in {'fastwam','bridgewam','boe'}:
        payload = {wrapper: payload}
    elif wrapper != 'native':
        payload = {k:v for k,v in payload.items() if k not in {'mot','proprio_encoder'}} | {
            wrapper: {f'module.fastwam.boe.{k}':v for k,v in original.bridge.state_dict().items()} |
                     {f'module.proprio_encoder.{k}':v for k,v in original.proprio_encoder.state_dict().items()}}
    torch.save(payload, path)
    restored = make_model()
    result = restored.load_checkpoint(path)
    assert result['step'] == 23
    assert_weights(original, restored)


def test_full_module_export_duplicates_are_checked(tmp_path):
    original = make_model()
    path = tmp_path/'fastwam.pt'
    payload = saved(original, path)
    payload.pop('mot'); payload.pop('proprio_encoder')
    payload['state_dict'] = original.state_dict()
    torch.save(payload, path)
    restored = make_model(); restored.load_checkpoint(path)
    assert_weights(original, restored)
    key = 'video_expert.blocks.0.modulation'
    payload['state_dict'][key] = payload['state_dict'][key] + 1
    with pytest.raises(ValueError, match='Conflicting'):
        normalize_checkpoint_payload(payload)


def test_reject_unknown_or_missing_weights_before_copy(tmp_path):
    model = make_model(); path=tmp_path/'incomplete.pt'
    payload = saved(model,path)
    for field in ['unexpected', 'missing']:
        candidate = copy.deepcopy(payload)
        if field == 'unexpected': candidate['mot']['unknown.weight'] = torch.ones(1)
        else: candidate['mot'].pop(next(k for k in candidate['mot'] if 'blocks.0' in k))
        torch.save(candidate, path)
        before = copy.deepcopy(model.state_dict())
        with pytest.raises(ValueError, match='Checkpoint state mismatch'):
            model.load_checkpoint(path)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before[key],value, rtol=0,atol=0)


def test_metaquery_rename_from_90ea435(tmp_path):
    model = make_model(); path=tmp_path/'MetaQueries_old.pt'
    payload = saved(model, path)
    payload['mot'] = {k.replace('latent_bridge_queries.lbq_embeddings','video_metaquery.query_embeddings').replace('mixtures.action.lbq_embedding.','mixtures.action.meta_embedding.'):v for k,v in payload['mot'].items()}
    meta = payload.pop('latent_bridge_queries')
    names={'num_lbqs':'num_queries','lbq_attention':'meta_attention','lbq_rope_mode':'meta_rope_mode','lbq_embedding_input_dim':'meta_embedding_input_dim','lbq_embedding_output_dim':'meta_embedding_output_dim'}
    payload['video_metaquery']={names.get(k,k):v for k,v in meta.items()}
    payload['video_metaquery']['injection_mode']='meta_only'
    payload['video_metaquery']['generation_coupling']='future_video_reads_meta'
    payload['action_dit_architecture']['conditioning_mode']='meta_only'
    payload['action_dit_architecture']['uses_meta_self_attention']=payload['action_dit_architecture'].pop('uses_lbq_self_attention')
    torch.save(payload,path)
    restored=make_model(); restored.load_checkpoint(path); assert_weights(model,restored)


def test_optimizer_state_resumes_after_name_only_change(tmp_path):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sum(p.square().sum() for p in model.parameters()).backward(); opt.step(); opt.zero_grad()
    path=tmp_path/'fastwam.pt'; model.save_checkpoint(path, optimizer=opt, step=4)
    payload=torch.load(path,weights_only=True); torch.save({'boe':payload},path)
    restored=make_model(); opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4)
    restored.load_checkpoint(path,optimizer=opt2)
    for m,o in ((model,opt),(restored,opt2)):
        sum(p.square().sum() for p in m.parameters()).backward();o.step()
    assert_weights(model,restored)


@pytest.mark.parametrize('change', ['multi_readout', 'bidirectional_action', 'shape'])
def test_incompatible_architecture_fails_before_copy(tmp_path, change):
    model=make_model(); path=tmp_path/'unsupported_boe.pt'; payload=saved(model,path)
    if change == 'multi_readout':
        payload['latent_bridge_queries']['readout_layer']=[0,1]
    elif change == 'bidirectional_action':
        payload['action_dit_architecture']['lbqs_read_action']=True
    else:
        payload['mot']['latent_bridge_queries.lbq_embeddings']=torch.ones(1)
    torch.save(payload,path)
    before=copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        model.load_checkpoint(path)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(before[key],value,rtol=0,atol=0)


def test_every_current_task_composes():
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        for task in (ROOT/'configs/task').glob('*.yaml'):
            cfg=compose(config_name='train',overrides=['task='+task.stem])
            assert cfg.model._target_.startswith('bridgewam.')
            OmegaConf.to_container(cfg,resolve=True)
