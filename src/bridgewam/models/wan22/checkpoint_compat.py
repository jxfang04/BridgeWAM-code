"""Name-only checkpoint migration. Never infer architecture from a filename.

Native 917/begin8.7 payloads use ``mot``/``mixtures``. The explicit wrapper
aliases below additionally support exported state dictionaries. Unknown keys
and conflicting aliases are errors, not silently discarded trained weights.
"""
from collections import OrderedDict
from collections.abc import Mapping

import torch


_ROOTS = ('module.', 'bridgewam.', 'fastwam.', 'boe.', 'mot.')
_MODULE_RENAMES = (
    ('experts.', 'mixtures.'),
    ('mixture.', 'mixtures.'),
    ('video_expert.', 'mixtures.video.'),
    ('action_expert.', 'mixtures.action.'),
    ('video_metaquery.', 'latent_bridge_queries.'),
)


def _strip_root(key):
    while key.startswith(_ROOTS):
        key = key.split('.', 1)[1]
    return key


def _mot_key(key):
    key = _strip_root(key)
    for old, new in _MODULE_RENAMES:
        if key.startswith(old):
            key = new + key[len(old):]
            break
    # Verified against commit 90ea435 (MetaQuery -> LBQ rename).
    key = key.replace('mixtures.action.meta_embedding.', 'mixtures.action.lbq_embedding.')
    if key in {'latent_bridge_queries.query_embeddings', 'latent_bridge_queries.meta_embeddings'}:
        key = 'latent_bridge_queries.lbq_embeddings'
    return key


def _same(left, right):
    if left is right:
        return True
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if torch.is_tensor(left) or torch.is_tensor(right):
        return False
    return left == right


def _put(target, key, value):
    if key in target:
        if not _same(target[key], value):
            raise ValueError(f'Conflicting checkpoint aliases for {key!r}.')
    target[key] = value


def _rename_state(state):
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint model state must be a mapping.')
    result = OrderedDict()
    for key, value in state.items():
        _put(result, _mot_key(key), value)
    if hasattr(state, '_metadata'):
        result._metadata = OrderedDict()
        for key, value in state._metadata.items():
            _put(result._metadata, _mot_key(key + '.').rstrip('.') if key else '', value)
    return result


def normalize_lbq_metadata(config):
    if not isinstance(config, Mapping):
        raise ValueError('LBQ checkpoint metadata must be a mapping.')
    names = {'num_queries': 'num_lbqs', 'meta_attention': 'lbq_attention',
             'meta_rope_mode': 'lbq_rope_mode', 'meta_embedding_input_dim': 'lbq_embedding_input_dim',
             'meta_embedding_output_dim': 'lbq_embedding_output_dim'}
    result = {}
    for key, value in config.items():
        key = names.get(key, key)
        if key == 'injection_mode':
            value = {'meta_only': 'lbq_only', 'text_state_meta': 'text_state_lbq'}.get(value, value)
        if key == 'generation_coupling' and value == 'future_video_reads_meta':
            value = 'future_video_reads_lbq'
        if key in result and result[key] != value:
            raise ValueError(f'Conflicting LBQ metadata aliases for {key}.')
        result[key] = value
    return result


def normalize_checkpoint_payload(payload):
    """Return canonical sections without mutating the caller's payload/tensors."""
    if not isinstance(payload, Mapping):
        raise ValueError('Checkpoint payload must be a mapping of weights and metadata.')
    result = dict(payload)
    # Named wrappers may contain a native payload or just its backbone state.
    for name in ('bridgewam', 'fastwam', 'boe'):
        if name not in result:
            continue
        candidate = result.pop(name)
        if not isinstance(candidate, Mapping):
            raise ValueError(f'Checkpoint section {name!r} must be a mapping.')
        if 'mot' in result:
            raise ValueError(f'Ambiguous checkpoint: both mot and {name} sections.')
        if 'mot' in candidate or 'dit' in candidate:
            for key, value in candidate.items():
                if key in result:
                    raise ValueError(f'Ambiguous nested checkpoint metadata: {key}.')
                result[key] = value
        else:
            result['mot'] = candidate
    if 'mot' not in result and 'dit' not in result:
        wrappers = [key for key in ('model_state_dict', 'state_dict', 'model') if key in result]
        if len(wrappers) > 1:
            raise ValueError(f'Ambiguous state-dict wrappers: {wrappers}.')
        if wrappers:
            flat = result.pop(wrappers[0])
        elif result and all(torch.is_tensor(v) for v in result.values()):
            flat, result = result, {}
        else:
            raise ValueError('Checkpoint requires mot/dit or a tensor state dictionary.')
        if not isinstance(flat, Mapping) or not flat or not all(torch.is_tensor(v) for v in flat.values()):
            raise ValueError('Exported state dictionary must contain tensors.')
        mot, proprio = OrderedDict(), OrderedDict()
        for key, value in flat.items():
            key = _strip_root(key)
            if key.startswith('dit.'):
                key = key[4:]
            if key.startswith('proprio_encoder.'):
                _put(proprio, key[len('proprio_encoder.'):], value)
                continue
            # Full Module.state_dict also includes the frozen base components.
            # They are loaded from pretrained component paths by the factory.
            if key.startswith(('vae.', 'text_encoder.')):
                continue
            key = _mot_key(key)
            # Keep all remaining keys, including buffers and unknown keys.
            # load_checkpoint validates them against the actual target model.
            _put(mot, key, value)
        if not mot:
            raise ValueError('Export contains no backbone weights.')
        result['mot'] = mot
        if proprio:
            if 'proprio_encoder' in result:
                combined = result['proprio_encoder'].copy()
                for key, value in proprio.items():
                    _put(combined, key, value)
                result['proprio_encoder'] = combined
            else:
                result['proprio_encoder'] = proprio
    if 'mot' in result:
        result['mot'] = _rename_state(result['mot'])
    if 'video_metaquery' in result:
        if 'latent_bridge_queries' in result:
            raise ValueError('Ambiguous video_metaquery/latent_bridge_queries metadata.')
        result['latent_bridge_queries'] = result.pop('video_metaquery')
    if result.get('latent_bridge_queries') is not None:
        result['latent_bridge_queries'] = normalize_lbq_metadata(result['latent_bridge_queries'])
    if isinstance(result.get('action_dit_architecture'), Mapping):
        arch = dict(result['action_dit_architecture'])
        if 'uses_meta_self_attention' in arch:
            if 'uses_lbq_self_attention' in arch:
                raise ValueError('Ambiguous ActionDiT attention metadata.')
            arch['uses_lbq_self_attention'] = arch.pop('uses_meta_self_attention')
        if 'conditioning_mode' in arch:
            arch['conditioning_mode'] = {'meta_only': 'lbq_only', 'text_state_meta': 'text_state_lbq'}.get(arch['conditioning_mode'], arch['conditioning_mode'])
        result['action_dit_architecture'] = arch
    return result
