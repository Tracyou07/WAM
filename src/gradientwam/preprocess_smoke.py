"""Explicit native preparation for one smoke episode or a disjoint episode split."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import subprocess
import sys
import yaml
from .settings import CAMERAS, load_episode_split


def commands(settings, *, source_root: Path, device: str, episode_ids=None):
    metadata=settings.preparation_root/'metadata'
    encode=[sys.executable,str(source_root/'scripts/pretraining/encoding/encode_latents.py'),
        '--dataset',str(settings.dataset_root),'--vae',str(settings.frontend_root/'vae'),
        '--out-root',str(settings.preparation_root/'latents'),'--latent-subdir','latents',
        '--fps','20','--size-mode','fixed','--resolution','128','--fit-mode','letterbox_pad',
        '--cameras',*CAMERAS,
        '--device',device,'--dtype','fp32' if device=='cpu' else 'bf16','--store-dtype','fp16',
        '--workers','1','--batch-size','1']
    condition=[sys.executable,str(source_root/'scripts/augment_lerobot_latents_with_single_frame_condition.py'),
        '--data-root',str(settings.dataset_root),'--video-root',str(settings.dataset_root/'videos'),
        '--reference-assets-root',str(settings.frontend_root),'--latent-root',str(settings.latent_root),
        '--device',device,'--batch-size','1','--output-dtype','float16','--source-frame-offset','-1',
        '--log-every','1']
    prompt=[sys.executable,str(source_root/'scripts/pretraining/text/encode_prompt_cache.py'),
        '--cfg',str(metadata/'prompt.yaml'),'--assets',str(settings.frontend_root),
        '--manifests',str(metadata/'prompts'),'--out',str(settings.prompt_root),
        '--batch-size','1','--device',device,'--allow-subset']
    # Native encoder accepts contiguous ranges, not an arbitrary JSON ID list.
    # Group consecutive selected IDs; never encode unselected gaps.
    ids = sorted(episode_ids if episode_ids is not None else [settings.episode_id])
    ranges = []
    for episode in ids:
        if ranges and episode == ranges[-1][0] + ranges[-1][1]:
            ranges[-1][1] += 1
        else:
            ranges.append([episode, 1])
    encoders = [encode + ['--start-episode',str(first),'--episodes',str(count)] for first,count in ranges]
    if episode_ids is None:
        condition += ['--max-files','1']
    return [*encoders,condition,prompt]


def prepare(settings, *, execute=False, device='cpu', episodes_file: Path | None = None):
    root=Path(__file__).resolve().parents[2]
    if not (root/'scripts').is_dir():
        raise RuntimeError('Preparation scripts require an editable install from the source checkout.')
    split = load_episode_split(episodes_file) if episodes_file is not None else None
    ids = sorted(split['train_episode_ids'] + split['heldout_episode_ids']) if split else [settings.episode_id]
    result={'status':'commands_only','commands':commands(settings,source_root=root,device=device,
                episode_ids=ids if split else None),
            'episode_ids':ids,'split':split,'preparation_root':str(settings.preparation_root)}
    if not execute:
        return result
    # A fresh output prevents accidental mixing of different encoder revisions.
    if settings.preparation_root.exists():
        raise FileExistsError('Preparation requires a fresh directory; reuse a completed cache with check-data/train.')
    info=json.loads((settings.dataset_root/'meta/info.json').read_text())
    if info.get('fps')!=20 or info.get('codebase_version')!='v2.1':
        raise ValueError('This preparation recipe requires LeRobot v2.1 at 20fps.')
    episodes=[json.loads(x) for x in (settings.dataset_root/'meta/episodes.jsonl').read_text().splitlines() if x]
    by_id = {e['episode_index']:e for e in episodes}
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f'Selected episodes absent from dataset metadata: {sorted(missing)}')
    tasks = set()
    for episode_id in ids:
        labels = by_id[episode_id].get('tasks')
        if not labels or any(not isinstance(t,str) or not t.strip() for t in labels):
            raise ValueError(f'Episode {episode_id} requires real task descriptions.')
        tasks.update(labels)
    metadata=settings.preparation_root/'metadata';manifests=metadata/'prompts'
    manifests.mkdir(parents=True,exist_ok=False)
    with (manifests/'selected_episode.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['task']);writer.writeheader()
        writer.writerows({'task':t} for t in sorted(tasks))
    if split:
        (metadata/'episodes.json').write_text(json.dumps(split,indent=2)+'\n',encoding='utf-8')
    native=dict(settings.native);native['data']=dict(native['data'])
    native['data']['empty_text_embedding_path']=None
    (metadata/'prompt.yaml').write_text(yaml.safe_dump(native),encoding='utf-8')
    for command in result['commands']:
        subprocess.run(command,cwd=root,check=True)
    index=json.loads((settings.prompt_root/'index.json').read_text())
    result.update(status='prepared_episode_split' if split else 'prepared_one_episode',
                  prompt_encoder_fingerprint=index['encoder_fingerprint'])
    (metadata/'preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
