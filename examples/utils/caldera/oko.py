"""Publish the baked Caldera scene to an OKO space through chs-tools.

  oko.py upload   (--space-id <id> | --global) [--only NAME ...] [--dry-run]
  oko.py entities [--out entities.json]

`upload` runs chs-tools' streamed-model-upload.ts once per model in scene.json and records the
root collection and GLB asset ids it prints in oko_ids.json (resumable: models already in the
file are skipped). `entities` turns scene.json plus those ids into an entities.json that
`space-entity-copy.ts import --keep-unmapped` can create the space from.

Auth is chs-tools' own: run `bin/mag-api-macos login` there first (CHS_TOOLS_ENV picks the environment).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CALDERA = os.path.join(HERE, '../../assets/meshlets/caldera')
SCENE = os.path.join(CALDERA, 'scene.json')
IDS = os.environ.get('OKO_IDS', os.path.join(HERE, 'oko_ids.json'))
CHS_TOOLS = os.environ.get('CHS_TOOLS', os.path.join(HERE, '../../../../chs-tools'))

ANSI = re.compile(r'\x1b\[[0-9;]*m')
STATIC_MODEL = 3


def load_scene():
    with open(SCENE) as f:
        return json.load(f)


def load_ids():
    if os.path.exists(IDS):
        with open(IDS) as f:
            return json.load(f)
    return {}


def model_name(model):
    return os.path.splitext(os.path.basename(model['url']))[0]


def upload(args):
    ids = load_ids()
    names = []
    for entity in load_scene()['entities']:
        for model in entity['models']:
            name = model_name(model)
            if name not in names:
                names.append(name)
    if args.only:
        names = [n for n in names if n in args.only]

    for i, name in enumerate(names, 1):
        if name in ids and not args.dry_run:
            print(f'[{i}/{len(names)}] {name}: already uploaded, skipping')
            continue
        print(f'[{i}/{len(names)}] {name}')
        target = ['--global'] if args.global_asset else ['--space-id', args.space_id]
        cmd = ['deno', 'run', '--allow-all', 'streamed-model-upload.ts', *target,
               '--glb', os.path.abspath(os.path.join(CALDERA, name + '.glb')), '--name', name]
        if args.dry_run:
            cmd.append('--dry-run')
        proc = subprocess.Popen(cmd, cwd=CHS_TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = []
        for line in proc.stdout:
            line = ANSI.sub('', line)
            out.append(line)
            if '[file]' not in line:
                print('   ' + line, end='')
        if proc.wait() != 0:
            sys.exit(f'{name}: upload failed (exit {proc.returncode}); rerun to resume')
        if args.dry_run:
            continue
        text = ''.join(out)
        collection = re.search(r'Root collection:\s*(\w+)', text)
        detail = re.search(r'Primary \(GLB\) detail:\s*(\w+)', text)
        if not collection or not detail:
            sys.exit(f'{name}: could not read the collection/asset ids from the uploader output')
        ids[name] = {'collectionId': collection.group(1), 'assetId': detail.group(1)}
        with open(IDS, 'w') as f:
            json.dump(ids, f, indent=2)


def entities(args):
    scene = load_scene()
    ids = load_ids()
    out = []
    for entity in scene['entities']:
        components = []
        for model in entity['models']:
            target = ids.get(model_name(model))
            if not target:
                sys.exit(f"{model_name(model)} has no uploaded ids in {IDS}; run `upload` first")
            components.append({
                'type': STATIC_MODEL,
                'typeName': 'StaticModel',
                'properties': {
                    '1': target['assetId'],
                    '2': target['collectionId'],
                    '3': model['position'],
                    '4': model['rotation'],
                    '5': model['scale'],
                    '6': model.get('visible', True),
                    '7': True,
                    '8': '',
                    '9': model.get('shadowCaster', True),
                    '10': {},
                    '11': True,
                    '12': False,
                    '13': False
                }
            })
        out.append({
            'id': entity['id'],
            'name': entity['name'],
            'position': entity['position'],
            'rotation': entity['rotation'],
            'scale': entity['scale'],
            'components': components,
            **({'parentId': entity['parentId']} if entity.get('parentId') is not None else {})
        })
    path = os.path.abspath(args.out)
    with open(path, 'w') as f:
        json.dump({
            'sourceSpaceId': 'caldera-scene-json',
            'exportedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'entities': out
        }, f, indent=2)
    print(f'{len(out)} entities -> {path}')


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest='command', required=True)
p = sub.add_parser('upload')
target = p.add_mutually_exclusive_group(required=True)
target.add_argument('--space-id')
target.add_argument('--global', dest='global_asset', action='store_true',
                    help='upload as global (space-less) assets, shown in the Global tab')
p.add_argument('--only', nargs='*', help='model names to upload, e.g. caldera_st_b')
p.add_argument('--dry-run', action='store_true')
p.set_defaults(fn=upload)
p = sub.add_parser('entities')
p.add_argument('--out', default=os.path.join(HERE, 'entities.json'))
p.set_defaults(fn=entities)
args = parser.parse_args()
args.fn(args)
