"""Retrieve the exact CSV export recorded in data/README.md."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/Zhelena/"
    "Analysis-of-Riboflavin-Production-in-Bacillus-subtilis/"
    "b3f2913180cb7df779ef10c4204486df64db6987/riboflavin.csv"
)
SHA256 = "41e7f85919fb2d355ae3a8c8bb0ea428ad74be06fe2f308e5c22324528f3cf0c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/riboflavin.csv"))
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Riboflavin source checksum does not match the recorded export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)


if __name__ == "__main__":
    main()
