"""Write generated data rows into the paper's table files, between marker comments, so the tables carry the numbers
inline. A table file marks a block with
    % rows: <fragment name>
    ...
    % end rows
(or a line \tabinput{staging/<fragment>.tex}, which is converted into such a block). Only the block is replaced; captions,
headers and anything else in the file stay as edited. Usage: python inline_tables.py <fragment dir> <table file>...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TAB = re.compile(r"^(\s*)\\tabinput\{staging/([\w.-]+?)(?:\.tex)?\}\s*$", re.M)
BLOCK = re.compile(r"^(\s*)% rows: ([\w.-]+)[^\n]*\n.*?^\s*% end rows\n", re.M | re.S)


def main() -> None:
    frags = Path(sys.argv[1])
    for path in map(Path, sys.argv[2:]):
        text = path.read_text()
        def block(indent: str, name: str) -> str:
            rows = (frags / f"{name}.tex").read_text().rstrip("\n").splitlines()
            return f"{indent}% rows: {name} (generated from the Vulcan runs; regenerated when results update)\n" + \
                   "".join(f"{indent}{r}\n" for r in rows) + f"{indent}% end rows\n"
        text = TAB.sub(lambda m: block(m[1], m[2]).rstrip("\n"), text)
        text = BLOCK.sub(lambda m: block(m[1], m[2].split(" ")[0]), text)
        path.write_text(text)
        print(path, "ok")


if __name__ == "__main__":
    main()
