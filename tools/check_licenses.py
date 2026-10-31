"""License allowlist check (guardrail S1).

Walks the installed dependency tree of handoff-check, plus any extras named on the command line,
and fails if a package's license is on the deny list or can't be identified.

    python tools/check_licenses.py            # core dependencies
    python tools/check_licenses.py hands      # core + the [hands] extra
"""

from __future__ import annotations

import re
import sys
from importlib import metadata

ALLOW = re.compile(r"\b(MIT|BSD|Apache|ISC|PSF|Python Software Foundation|MPL|Mozilla|LGPL|Unlicense|HPND|Zlib|0BSD|CC0)", re.I)
DENY = re.compile(r"\b(AGPL|Affero|non-?commercial|CC[- ]BY[- ]NC|research (only|license)|proprietary)\b", re.I)
GPL_ONLY = re.compile(r"(?<!L)GPL", re.I)


def _name(req: str) -> str:
    return re.split(r"[\s;<>=!~\[]", req, maxsplit=1)[0].lower()


def license_of(dist: metadata.Distribution) -> str:
    md = dist.metadata
    parts = [md.get("License-Expression") or "", md.get("License") or ""]
    parts += [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    text = " | ".join(p.strip() for p in parts if p and p.strip() and len(p) < 200)
    if not text and md.get("License"):
        text = md.get("License").splitlines()[0][:120]  # full license text pasted into the field
    return text or "UNKNOWN"


def dependency_tree(root: str, extras: set[str]) -> dict[str, metadata.Distribution]:
    seen: dict[str, metadata.Distribution] = {}
    stack = [(root, extras)]
    while stack:
        name, wanted = stack.pop()
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        key = dist.metadata["Name"].lower()
        if key in seen and not wanted:
            continue
        seen[key] = dist
        for req in dist.requires or []:
            marker = req.split(";", 1)[1] if ";" in req else ""
            extra = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker)
            if extra and extra.group(1) not in wanted:
                continue
            # Platform markers are fine to ignore: packages that aren't installed are skipped above.
            stack.append((_name(req), set()))
    seen.pop(root.lower(), None)
    return seen


def main(argv: list[str]) -> int:
    extras = set(argv)
    tree = dependency_tree("handoff-check", extras)
    if not tree:
        print("handoff-check is not installed (run: pip install -e .)", file=sys.stderr)
        return 2
    bad = 0
    for key in sorted(tree):
        lic = license_of(tree[key])
        if DENY.search(lic) or (GPL_ONLY.search(lic) and not ALLOW.search(lic)):
            status, bad = "DENIED", bad + 1
        elif ALLOW.search(lic):
            status = "ok"
        else:
            status, bad = "CHECK", bad + 1
        print(f"{status:7} {key:28} {lic}")
    if bad:
        print(f"\n{bad} package(s) need a look before release.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
