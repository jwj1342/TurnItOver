"""离线合并两种 generator 的 Table 1 批处理结果；不调用模型。"""
import argparse
import json
from pathlib import Path

from turnitover.repair.table1_scoring import summarize_table1


def summarize(paths, generators):
    rows = []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        rows.extend(report['results'])
    return summarize_table1(rows, generators)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', type=Path, nargs='+')
    parser.add_argument('--generators', nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.summaries, args.generators)
    # Preserve previous analyses; never overwrite a filled table accidentally.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({'conditions': len(result), 'output': str(args.output)}))
