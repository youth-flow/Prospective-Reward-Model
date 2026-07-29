from __future__ import annotations

import json
import re
import sys
from typing import Any


def extract_revision(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Apptainer inspect payload must be an object")
    labels = payload.get("data", {}).get("attributes", {}).get("labels", {})
    revision = labels.get("org.opencontainers.image.revision")
    if not isinstance(revision, str):
        raise ValueError("image revision label is missing")

    # Apptainer preserves quotes written around %labels values as label content.
    if revision.startswith('"') and revision.endswith('"'):
        revision = json.loads(revision)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("image revision label is not a 40-character Git commit")
    return revision


def main() -> None:
    try:
        print(extract_revision(json.load(sys.stdin)))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
