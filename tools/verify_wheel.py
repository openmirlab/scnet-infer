"""Install the latest wheel into a throwaway environment and touch the facade.

This catches empty wheel-from-sdist builds and missing packaged checkpoint data.
Reads: dist/*.whl and the uv executable.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    wheels = sorted((ROOT / "dist").glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise SystemExit("No wheel found in dist; run `uv run python -m build` first")
    wheel = wheels[-1]
    with tempfile.TemporaryDirectory(prefix="scnet-infer-wheel-") as temporary:
        environment = Path(temporary) / "venv"
        subprocess.run(["uv", "venv", str(environment)], check=True)
        python = environment / "bin" / "python"
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
        subprocess.run(
            [str(python), "-c", "import scnet_infer; from scnet_infer.checkpoints import load_manifest; assert scnet_infer.SCNetSession; assert load_manifest()[0]"],
            check=True,
        )
    print(f"Wheel verification passed: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

