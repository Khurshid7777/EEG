"""Write requirements.lock: the pinned closure of the release dependencies (core + api + lsl) as
installed in the environment where the test suite and the pre-registered checks ran (D85).

    python scripts/make_lock.py > requirements.lock
    pip-audit -r requirements.lock --no-deps --disable-pip -f json -o docs/regulatory/soup_audit.json
"""

import importlib.metadata as m
import re
import sys

ROOTS = ["numpy", "scipy", "pyedflib", "networkx", "pydantic", "pyyaml", "jsonschema", "matplotlib",
         "reportlab", "pillow", "fastapi", "uvicorn", "python-multipart", "pylsl"]


def closure(roots: list[str]) -> dict[str, tuple[str, str]]:
    seen, todo = {}, list(roots)
    while todo:
        name = todo.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        try:
            d = m.distribution(name)
        except m.PackageNotFoundError:
            continue
        seen[key] = (d.metadata["Name"], d.version)
        todo += [re.split(r"[ ;<>=!~\[(]", r, maxsplit=1)[0] for r in d.requires or [] if "extra ==" not in r]
    return seen


def main() -> int:
    print("# Pinned release dependencies (core + api + lsl): the environment of the test suite and the")
    print(f"# pre-registered checks (Python {sys.version.split()[0]}). Regenerate: scripts/make_lock.py (D85).")
    for _, (name, ver) in sorted(closure(ROOTS).items()):
        print(f"{name}=={ver}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
