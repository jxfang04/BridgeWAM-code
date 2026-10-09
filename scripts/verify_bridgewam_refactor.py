"""Compare BridgeWAM with an explicitly selected Git revision using tiny CPU models."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-repo",
        required=True,
        type=Path,
        help="Full repository containing the reference implementation.",
    )
    parser.add_argument(
        "--reference-ref",
        required=True,
        help="Explicit Git revision of the reference implementation to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for numerical fixtures; otherwise temporary.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts/validation/reference_worker.py"
    with tempfile.TemporaryDirectory(prefix="bridgewam-reference-") as temp:
        temp = Path(temp)
        reference = temp / "source"
        reference.mkdir()
        archive = temp / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(archive), "--", args.reference_ref, "src"],
            cwd=args.reference_repo,
            check=True,
        )
        subprocess.run(["tar", "-xf", str(archive), "-C", str(reference)], check=True)
        fixtures = args.output_dir.resolve() if args.output_dir else temp / "fixtures"
        for mode, source in [("reference", reference), ("current", root)]:
            env = dict(
                os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source / "src")
            )
            subprocess.run(
                [sys.executable, str(worker), mode, str(fixtures)],
                cwd=temp,
                env=env,
                check=True,
            )
    print(
        "All 12 numerical cases match exactly on CPU. Full GPU/checkpoint/simulator validation is separate."
    )


if __name__ == "__main__":
    main()
