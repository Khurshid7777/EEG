"""Read-only access to BDSP data in AWS S3 through access points (owner-approved use of the owner's own read-only keys).

Keys are read ONLY from BDSP_AWS_ACCESS_KEY_ID / BDSP_AWS_SECRET_ACCESS_KEY (never from AWS_*: the sandbox's own AWS
variables are not the owner's and are not used). Nothing is printed except identities, key names and sizes.

    python scripts/bdsp_s3.py check
    python scripts/bdsp_s3.py ls  ARN [PREFIX] [--max N]
    python scripts/bdsp_s3.py get ARN KEY OUT_FILE
    # open-access point:  arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-open-access-point
"""

import os
import sys
from pathlib import Path

REGION = "us-east-1"
OPEN_ARN = "arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-open-access-point"


def _client(service="s3"):
    import boto3

    key, secret = os.environ.get("BDSP_AWS_ACCESS_KEY_ID"), os.environ.get("BDSP_AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise SystemExit("BDSP_AWS_ACCESS_KEY_ID / BDSP_AWS_SECRET_ACCESS_KEY are not set (AWS_* variables are deliberately ignored).")
    return boto3.client(service, region_name=REGION, aws_access_key_id=key, aws_secret_access_key=secret,
                        aws_session_token=os.environ.get("BDSP_AWS_SESSION_TOKEN"))


def check() -> int:
    ident = _client("sts").get_caller_identity()
    print({"account": ident["Account"], "arn": ident["Arn"]})
    try:
        r = _client().list_objects_v2(Bucket=OPEN_ARN, MaxKeys=5, Delimiter="/")
        print("open access point listing ok:", [c["Prefix"] for c in r.get("CommonPrefixes", [])][:5])
    except Exception as exc:  # noqa: BLE001
        print("open access point listing failed:", type(exc).__name__, str(exc)[:200])
    return 0


def ls(arn: str, prefix: str = "", max_keys: int = 50) -> int:
    c, token, n = _client(), None, 0
    while n < max_keys:
        kw = {"Bucket": arn, "Prefix": prefix, "MaxKeys": min(1000, max_keys - n), "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        r = c.list_objects_v2(**kw)
        for p in r.get("CommonPrefixes", []):
            print("DIR ", p["Prefix"]); n += 1
        for o in r.get("Contents", []):
            print(f"{o['Size']:>12} {o['Key']}"); n += 1
        token = r.get("NextContinuationToken")
        if not token:
            break
    return 0


def get(arn: str, key: str, out: str) -> int:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(arn, key, out)
    print("saved", out, Path(out).stat().st_size)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "check":
        sys.exit(check())
    if a[0] == "ls":
        sys.exit(ls(a[1], a[2] if len(a) > 2 and not a[2].startswith("--") else "", int(a[a.index("--max") + 1]) if "--max" in a else 50))
    if a[0] == "get":
        sys.exit(get(a[1], a[2], a[3]))
    raise SystemExit(__doc__)
