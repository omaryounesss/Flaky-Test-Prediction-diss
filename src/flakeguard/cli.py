"""flakeguard CLI: score a repo's JUnit tests for flakiness risk.

    flakeguard train --out models/flakeguard.joblib
    flakeguard scan path/to/repo --model models/flakeguard.joblib --top 20

`train` fits a TF-IDF + XGBoost model on the FlakeFlagger dataset's per-test
token lists. `scan` extracts @Test methods from Java source, tokenizes their
bodies the same way, and ranks them by predicted flakiness risk; risky-API
detectors provide the human-readable reasons. Without a model file, `scan`
falls back to a pure heuristic score (share of risky-API families present).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .extract import RISKY_APIS

JAVA_KEYWORDS = frozenset(
    """abstract assert boolean break byte case catch char class const continue
    default do double else enum extends final finally float for goto if
    implements import instanceof int interface long native new package private
    protected public return short static strictfp super switch synchronized
    this throw throws transient try void volatile while true false null""".split()
)

CAMEL = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_token_list(s: str) -> list[str]:
    """TF-IDF tokenizer for comma-joined token strings (module-level so the
    fitted pipeline can be pickled)."""
    return [t for t in s.split(",") if t]


def tokenize_java(body: str) -> str:
    """Identifier tokens, comma-joined, matching the dataset's tokenList
    conventions: whole identifiers lowercased (e.g. assertEquals ->
    'assertequals') plus their camelCase parts, keywords dropped."""
    tokens = []
    for ident in IDENT.findall(body):
        whole = ident.lower()
        if len(whole) > 1 and whole not in JAVA_KEYWORDS:
            tokens.append(whole)
        parts = [p.lower() for p in CAMEL.findall(ident)]
        if len(parts) > 1:
            tokens.extend(p for p in parts
                          if len(p) > 1 and p not in JAVA_KEYWORDS)
    return ",".join(tokens)


def cmd_train(args: argparse.Namespace) -> int:
    import joblib
    from sklearn.pipeline import Pipeline

    from .data import load_vocabulary
    from .modeling import make_model, pos_weight
    from .vocab import _vectorizer

    frame = load_vocabulary()
    y = frame["flaky"].values
    model = Pipeline([
        ("tfidf", _vectorizer()),
        ("clf", make_model("xgboost", pos_weight=pos_weight(y))),
    ])
    model.fit(frame["tokenList"].values, y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"trained on {len(frame)} tests ({int(y.sum())} flaky) -> {out}")
    return 0


def _heuristic_score(features: dict[str, float]) -> float:
    hits = sum(features.get(k, 0.0) for k in RISKY_APIS)
    return hits / len(RISKY_APIS)


def cmd_scan(args: argparse.Namespace) -> int:
    from .extract import scan_repo

    root = Path(args.path)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2
    tests = scan_repo(root)
    if not tests:
        print("no @Test methods found")
        return 0

    model = None
    if args.model and Path(args.model).exists():
        import joblib

        model = joblib.load(args.model)

    if model is not None:
        scores = model.predict_proba([tokenize_java(t.body) for t in tests])[:, 1]
    else:
        scores = [_heuristic_score(t.features) for t in tests]

    ranked = sorted(zip(scores, tests), key=lambda p: -p[0])[: args.top]
    results = [
        {
            "test": f"{t.file}::{t.name}",
            "risk": round(float(s), 3),
            "reasons": t.reasons or ["no specific risky pattern detected"],
        }
        for s, t in ranked
    ]

    if args.format == "json":
        print(json.dumps({"scored": len(tests), "results": results}, indent=2))
    elif args.format == "markdown":
        print(f"## Flakeguard report — {len(tests)} tests scored\n")
        print("| Risk | Test | Why |")
        print("|---|---|---|")
        for r in results:
            print(f"| {r['risk']:.2f} | `{r['test']}` | {'; '.join(r['reasons'])} |")
    else:
        mode = "model" if model is not None else "heuristic (no model file)"
        print(f"flakeguard: scored {len(tests)} tests [{mode}]\n")
        for r in results:
            print(f"  {r['risk']:.2f}  {r['test']}")
            for reason in r["reasons"]:
                print(f"        - {reason}")

    if args.fail_above is not None and results and results[0]["risk"] > args.fail_above:
        print(f"\nrisk gate: top score {results[0]['risk']:.2f} > {args.fail_above}",
              file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flakeguard")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the token model on the dataset")
    p_train.add_argument("--out", default="models/flakeguard.joblib")
    p_train.set_defaults(func=cmd_train)

    p_scan = sub.add_parser("scan", help="score a repo's tests for flakiness risk")
    p_scan.add_argument("path")
    p_scan.add_argument("--model", default="models/flakeguard.joblib")
    p_scan.add_argument("--top", type=int, default=20)
    p_scan.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    p_scan.add_argument("--fail-above", type=float, default=None,
                        help="exit 1 if any test scores above this risk (CI gate)")
    p_scan.set_defaults(func=cmd_scan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
