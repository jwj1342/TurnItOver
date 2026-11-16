"""Serve the frozen Ordinary 2B checkpoint in a separately allocated GPU environment."""
import argparse
from pathlib import Path
import sys
from repair import check_sources, read, write, sha, HERE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'model', 'checkpoint', 'training-record'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    paths = read(a.work / 'inputs.private.json')
    identity = check_sources(Path(paths['runtime']))
    # The original GPU service does not import the repair/browser dependency stack.
    sys.path.insert(0, paths['runtime'])
    expected = next(r['sha256'] for r in identity['frozen_files'] if r['path'] == 'frozen/infer_server.py')
    if sha(a.work / 'code-v2/infer_server.py') != expected:
        raise ValueError('Prepared inference source changed')
    record = read(a.training_record)
    if not (record.get('status') == 'complete' and record.get('supervision') == 'ordinary'
            and record.get('seed') == 20261013 and record.get('examples') == 600
            and record.get('dataset_manifest_sha256') == identity['dataset_manifest_sha256']):
        raise ValueError('Requires completed v1.1 Ordinary seed 20261013 training record')
    if sha(a.model / 'config.json') != record['base_model_config_sha256']:
        raise ValueError('Base model config does not match training record')
    if not (a.checkpoint / 'adapter_model.safetensors').is_file():
        raise ValueError('Adapter weights are missing')
    queue = a.work / 'ordinary-queue-v2'
    queue.mkdir(exist_ok=True)
    if (queue / 'stop').exists() or (queue / 'ready.json').exists():
        raise ValueError('Queue has a prior server lifecycle; inspect before restarting')
    write(queue / 'checkpoint-identity.private.json', {'training_record_sha256': sha(a.training_record),
          'adapter_sha256': sha(a.checkpoint / 'adapter_model.safetensors'), 'base_config_sha256': sha(a.model / 'config.json')})
    # Only path selection is adapted; tokenization, generation and serving loop stay exact.
    script = (a.work / 'code-v2/infer_server.py').read_text(encoding='utf-8')
    old_model = "basepath=a.root/'Qwen3-VL-2B-Instruct'"
    old_adapter = "a.root/'runs/ordinary-20261013/checkpoint'"
    if script.count(old_model) != 1 or script.count(old_adapter) != 1:
        raise ValueError('Unexpected frozen inference source')
    script = script.replace(old_model, 'basepath=MODEL_PATH').replace(old_adapter, 'ADAPTER_PATH')
    sys.argv = ['infer_server.py', str(a.work), str(queue)]
    exec(compile(script, str(a.work / 'code-v2/infer_server.py'), 'exec'),
         {'__name__': '__main__', 'MODEL_PATH': a.model.resolve(), 'ADAPTER_PATH': a.checkpoint.resolve()})


if __name__ == '__main__':
    main()
