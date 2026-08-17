#!/usr/bin/env python3
"""Model-family chart + heuristic detector for GGUF files.

Prints the chart of which detection heuristics map to which model family, or
(if given a GGUF) the ranked heuristic guesses for that specific file --
useful when convert.py reports an unrecognized general.architecture, to see
what *close enough* family it would fall back to and how to force it.

Usage:
  venv/bin/python tools/detect_family.py --chart
  venv/bin/python tools/detect_family.py <model.gguf>
  venv/bin/python tools/detect_family.py <model.gguf> --json
"""

import argparse
import json
import sys
from pathlib import Path

from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from q4nx.arch_detect import (  # noqa: E402
    FAMILY_PROFILES,
    detect_model_family,
    render_chart,
)
from q4nx.constants import ModelArchNames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", nargs="?", help="GGUF file to detect the family of")
    parser.add_argument(
        "--chart", action="store_true", help="print the family -> heuristic chart"
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable detection output"
    )
    args = parser.parse_args()

    if args.chart or args.gguf is None:
        print(render_chart())
        return 0

    reader = GGUFReader(args.gguf)
    guesses = detect_model_family(reader)

    if args.json:
        payload = {
            "file": args.gguf,
            "guesses": [
                {
                    "arch": g.arch.name,
                    "arch_names": ModelArchNames.get(g.arch, []),
                    "confidence": g.confidence,
                    "score": g.score,
                    "reasons": g.reasons,
                }
                for g in guesses
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    arch_str = None
    basename = None
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch_str = str(field.parts[field.data[0]], encoding="utf-8") if field.data else None
        elif field.name in ("general.basename", "general.name") and basename is None:
            basename = str(field.parts[field.data[0]], encoding="utf-8") if field.data else None

    print(f"file: {args.gguf}")
    print(f"  general.architecture : {arch_str or '(missing)'}")
    print(f"  basename/name        : {basename or '(missing)'}")
    print(f"  tensors              : {len(reader.tensors)}")
    if not guesses:
        print("  => no family matched (nothing close enough). Use -f to force.")
        return 1
    print(f"  => top guess: {guesses[0].arch.name} ({guesses[0].confidence})")
    for i, guess in enumerate(guesses, 1):
        print(f"\n  #{i} {guess.arch.name}  [{guess.confidence} confidence, score {guess.score}]")
        for reason in guess.reasons:
            print(f"      - {reason}")
    print("\n  (in convert.py a registered top guess is taken with a warning;")
    print("   force a specific type with:  -f " + guesses[0].arch.name + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
