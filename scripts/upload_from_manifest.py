#!/usr/bin/env python3
"""
Commit (and optionally push) one file per REPO_MANIFEST.json entry.

Each commit message = the unique description for that path, so GitHub shows
distinct "Last commit message" per file.

Usage (repo root):
  python scripts/upload_from_manifest.py              # commit all, push at end
  python scripts/upload_from_manifest.py --dry-run    # preview only
  python scripts/upload_from_manifest.py --push-each  # push after every file (slow)
  python scripts/upload_from_manifest.py --from 50    # resume from index 50
  python scripts/upload_from_manifest.py --new-only   # only untracked/changed files
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "REPO_MANIFEST.json"
REMOTE_BRANCH = "main"
LOCAL_BRANCH = None  # current branch


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def _is_tracked(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def _has_changes(path: str) -> bool:
    if not _is_tracked(path):
        return True
    if subprocess.run(["git", "diff", "--quiet", "--", path], cwd=ROOT).returncode != 0:
        return True
    return (
        subprocess.run(["git", "diff", "--cached", "--quiet", "--", path], cwd=ROOT).returncode
        != 0
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="One commit per REPO_MANIFEST file.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only.")
    ap.add_argument("--push-each", action="store_true", help="Push after each commit.")
    ap.add_argument("--no-push", action="store_true", help="Skip final push.")
    ap.add_argument("--from", dest="start_from", type=int, default=0, metavar="N",
                    help="Start at manifest index N (0-based, for resume).")
    ap.add_argument("--new-only", action="store_true",
                    help="Skip files already committed with no local changes.")
    args = ap.parse_args()

    if not MANIFEST.is_file():
        print(f"Missing {MANIFEST}", file=sys.stderr)
        return 1

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files: dict[str, str] = payload.get("files") or {}

    # Ensure manifest itself is committed (with catalog description).
    extra = {
        "REPO_MANIFEST.json": "Machine-readable map: every tracked path → unique description.",
    }
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, desc in {**files, **extra}.items():
        if path not in seen:
            ordered.append((path, desc))
            seen.add(path)
    # Stable order: REPO_MANIFEST first, then manifest file order, extras merged above
    # Re-build: REPO_MANIFEST first, then all keys from files in JSON order
    ordered = [("REPO_MANIFEST.json", extra["REPO_MANIFEST.json"])]
    for path, desc in files.items():
        ordered.append((path, desc))

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()

    total = len(ordered)
    ok = skip = fail = 0

    for i, (path, message) in enumerate(ordered):
        if i < args.start_from:
            continue
        fp = ROOT / path
        if not fp.is_file():
            print(f"[{i+1}/{total}] SKIP missing: {path}")
            skip += 1
            continue

        if args.new_only and not _has_changes(path):
            print(f"[{i+1}/{total}] SKIP unchanged: {path}")
            skip += 1
            continue

        print(f"\n[{i+1}/{total}] {path}")
        if args.dry_run:
            preview = message[:80] + ("..." if len(message) > 80 else "")
            try:
                print(f"  commit: {preview}")
            except UnicodeEncodeError:
                print(f"  commit: {preview.encode('ascii', 'replace').decode()}")
            ok += 1
            continue

        try:
            run(["git", "add", "--", path])
            cp = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                if "nothing to commit" in (cp.stdout + cp.stderr):
                    print("  SKIP: nothing to commit")
                    skip += 1
                    continue
                print(cp.stdout, cp.stderr, file=sys.stderr)
                raise subprocess.CalledProcessError(cp.returncode, cp.args)
            ok += 1
            if args.push_each:
                run(["git", "push", "origin", f"{branch}:{REMOTE_BRANCH}"])
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            fail += 1
            return 1

    if not args.dry_run and not args.no_push and not args.push_each:
        print("\nPushing all commits to GitHub...")
        run(["git", "push", "origin", f"{branch}:{REMOTE_BRANCH}"])

    print(f"\nDone: committed={ok}, skipped={skip}, failed={fail}, total={total}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
