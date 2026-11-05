"""Reproduce rejected causal weight edits without fitting or graph changes.

These are research controls, not model upgrades. Every candidate retains the
source 6→128→128→3 Tanh network. See the causal-weight-audit-v4 results.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import onnx
import torch

from competition_eval import load_mlp
from policy import ROOT, export


SOURCE = ROOT/'training/results/pace-study/training/candidate/pacer/best.pt'
RUN = ROOT/'training/results/pace-study/training/dagger-continuation-s412'


def candidates():
    source = load_mlp(SOURCE)
    result = {}
    for gain in (.9, 1.1):
        model = copy.deepcopy(source)
        group = [34, 62, 68]
        constants = torch.tensor([-1., 1., -1.])
        with torch.no_grad():
            columns = source[-1].weight[:, group].clone()
            model[-1].weight[:, group] = gain*columns
            model[-1].bias += (1-gain)*(columns@constants)
        result[f'centered-gain-{gain}'] = model
    for name, unit, earlier, constant in [
        ('restore-up-96-r24', 96, 24, -1.),
        ('restore-down-91-r18', 91, 18, 1.),
    ]:
        donor = load_mlp(RUN/f'round-{earlier}.pt')
        model = copy.deepcopy(source)
        with torch.no_grad():
            model[2].weight[unit].copy_(donor[2].weight[unit])
            model[2].bias[unit].copy_(donor[2].bias[unit])
            old_column = source[-1].weight[:, unit]
            new_column = donor[-1].weight[:, unit]
            model[-1].weight[:, unit].copy_(new_column)
            model[-1].bias += constant*(old_column-new_column)
        result[name] = model
    model = copy.deepcopy(source)
    with torch.no_grad():
        model[-1].weight[1, 89] -= .5
        model[-1].bias[1] += .5
    result['phase-braking'] = model
    return source, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh directory; preserve the rejected experiments.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    source, models = candidates()
    result = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        purpose='Rejected causal controls; no model improvement established', models={})
    code = """import{readFileSync}from'node:fs';import{validateArenaModel}from'./src/validate.ts';
const s=validateArenaModel(readFileSync(process.argv[1]));
console.log(JSON.stringify({params:s.params,macs:s.macs,nodes:s.nodes,widths:s.architecture.widths}));"""
    for name, model in models.items():
        path = args.out/(name+'.onnx')
        export(model, path)
        graph = onnx.load(path)
        onnx.helper.set_model_props(graph, dict(rules='pong-v4',
            contract='observation-6/action-scores-3/dense-mlp-v2',
            architecture='6-128-128-3 Tanh',
            method='Causal numerical weight edit or aligned learned-feature transfer; no training',
            status='Rejected research control; not a demonstrated upgrade'))
        onnx.save(graph, path)
        changed = {}
        for key, value in model.state_dict().items():
            before = source.state_dict()[key]
            count = int((value != before).sum())
            if count:
                changed[key] = count
        stats = json.loads(subprocess.run(['node', '--import', 'tsx', '--input-type=module',
            '-e', code, str(path)], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
        assert stats['widths'] == [6, 128, 128, 3] and stats['nodes'] == 5
        result['models'][name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            changed_parameters=changed, **stats)
    (args.out/'models.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
