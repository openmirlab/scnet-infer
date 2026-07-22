"""Verify the shipped package boundary contains inference code only.

The scan checks Python definitions/imports and dependency metadata rather than
flagging attribution prose or fixture-generation tools outside the wheel.
Reads: src/scnet_infer and pyproject.toml.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "scnet_infer"
FORBIDDEN_DEPS = {"accelerate", "lightning", "pytorch-lightning", "tensorboard", "wandb"}
FORBIDDEN_NAMES = re.compile(r"^(train|training_step|fit|evaluate|dataset|dataloader|optimizer)$", re.I)


def main() -> int:
    failures: list[str] = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and FORBIDDEN_NAMES.match(node.name):
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: forbidden shipped symbol {node.name}")
            if isinstance(node, ast.Import):
                modules = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split('.')[0]]
            else:
                modules = []
            for module in modules:
                if module.lower() in FORBIDDEN_DEPS:
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: forbidden dependency {module}")
    metadata = (ROOT / "pyproject.toml").read_text().lower()
    for dependency in FORBIDDEN_DEPS:
        if re.search(rf'^[ \t]*"{re.escape(dependency)}(?:[<>=~!]|\")', metadata, re.M):
            failures.append(f"pyproject.toml: forbidden dependency {dependency}")
    if failures:
        print("Inference-only verification failed:")
        print("\n".join(failures))
        return 1
    print(f"Inference-only verification passed: scanned {len(list(SOURCE.rglob('*.py')))} shipped Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

