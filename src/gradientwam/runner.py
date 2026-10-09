"""Public entrypoint. Checking config never builds a model or opens CUDA."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .settings import load_settings, load_episode_split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check-config','check-data','prepare','train'))
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--resume',type=Path,help='Matching step1 full checkpoint; resume to step2 only.')
    parser.add_argument('--execute',action='store_true',help='Run preparation commands; default only prints them.')
    parser.add_argument('--device',default='cpu',help='Preparation device only; training uses one selected CUDA device.')
    parser.add_argument('--episodes-file',type=Path,help='Versioned train_episode_ids/heldout_episode_ids split JSON for prepare/check-data.')
    args=parser.parse_args()
    if args.episodes_file is not None and args.command not in ('prepare','check-data'):
        parser.error('--episodes-file applies to prepare/check-data; distributed training has its own split argument.')
    settings=load_settings(args.config)
    if args.command=='check-config':
        settings.native_config()
        result={'status':'config_valid','arm':settings.arm,'asset_existence_checked':False,
                'model_constructed':False,'native_identity':settings.identity()}
    elif args.command=='check-data':
        from .data_check import load_sample
        if args.episodes_file is None:
            _,result=load_sample(settings,settings.native_config())
        else:
            split=load_episode_split(args.episodes_file)
            result={'status':'split_data_checked','split':split,'samples':[],'control_evaluation':False}
            for partition in ('train_episode_ids', 'heldout_episode_ids'):
                ids=split[partition]
                for episode in ids:
                    selected=replace(settings,episode_id=episode)
                    _,report=load_sample(selected,selected.native_config())
                    result['samples'].append({'partition':partition,**report})
    elif args.command=='prepare':
        from .preprocess_smoke import prepare
        result=prepare(settings,execute=args.execute,device=args.device,episodes_file=args.episodes_file)
    else:
        from .train_smoke import run
        result=run(settings,resume=args.resume)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
