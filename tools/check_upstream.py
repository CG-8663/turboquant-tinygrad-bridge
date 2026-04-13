#!/usr/bin/env python3
"""Check upstream dependencies for changes since last submodule pin.

Usage:
    python tools/check_upstream.py
    python tools/check_upstream.py --fetch     # fetch remotes first
    python tools/check_upstream.py --verbose   # show commit details
"""

import argparse
import os
import subprocess
import sys
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_YAML = os.path.join(ROOT, "upstream.yaml")

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def run(cmd, cwd=None):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or ROOT)
    return result.stdout.strip(), result.returncode


def check_submodule(name, config, fetch=False, verbose=False):
    submodule = config.get("submodule")
    if not submodule:
        return check_remote_only(name, config, fetch, verbose)

    sub_path = os.path.join(ROOT, submodule)
    if not os.path.isdir(sub_path):
        print(f"  {RED}NOT FOUND{RESET} — {sub_path}")
        return 1

    if fetch:
        run(["git", "fetch", "origin"], cwd=sub_path)

    track = config.get("track_branch", "main")
    pinned, _ = run(["git", "rev-parse", "HEAD"], cwd=sub_path)
    latest, rc = run(["git", "rev-parse", f"origin/{track}"], cwd=sub_path)

    if rc != 0:
        print(f"  {YELLOW}UNKNOWN{RESET} — can't resolve origin/{track} (try --fetch)")
        return 0

    if pinned == latest:
        print(f"  {GREEN}UP TO DATE{RESET} — pinned at {pinned[:10]}")
        return 0

    # Count commits behind
    behind, _ = run(
        ["git", "rev-list", "--count", f"HEAD..origin/{track}"],
        cwd=sub_path,
    )

    print(f"  {YELLOW}{behind} commits behind{RESET} origin/{track}")
    print(f"  Pinned:  {pinned[:10]}")
    print(f"  Latest:  {latest[:10]}")

    if verbose:
        # Check watched paths for changes
        watch = config.get("watch_paths", [])
        if watch:
            pathspecs = [p.rstrip("/") for p in watch]
            diff_cmd = ["git", "diff", "--stat", "HEAD", f"origin/{track}", "--"] + pathspecs
            diff_out, _ = run(diff_cmd, cwd=sub_path)
            if diff_out:
                print(f"  {BOLD}Changed watched files:{RESET}")
                for line in diff_out.split("\n")[:10]:
                    print(f"    {line}")
            else:
                print(f"  {DIM}No changes in watched paths{RESET}")

        # Recent commits
        log_out, _ = run(
            ["git", "log", "--oneline", "-5", f"HEAD..origin/{track}"],
            cwd=sub_path,
        )
        if log_out:
            print(f"  {BOLD}Recent upstream commits:{RESET}")
            for line in log_out.split("\n"):
                print(f"    {line}")

    return int(behind) if behind.isdigit() else 1


def check_remote_only(name, config, fetch=False, verbose=False):
    """For dependencies without a submodule (e.g., upstream llama.cpp)."""
    remote = config.get("remote", "")
    track = config.get("track_branch", "main")

    if not remote:
        print(f"  {DIM}No remote configured{RESET}")
        return 0

    if fetch:
        # Use ls-remote to check latest without cloning
        out, rc = run(["git", "ls-remote", "--heads", remote, track])
        if rc == 0 and out:
            sha = out.split()[0]
            print(f"  {GREEN}Latest:{RESET} {sha[:10]} on {track}")
        else:
            print(f"  {YELLOW}Can't reach{RESET} {remote}")
    else:
        print(f"  {DIM}Remote-only — use --fetch to check{RESET}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Check upstream dependencies")
    parser.add_argument("--fetch", action="store_true", help="Fetch remotes first")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show commit details")
    args = parser.parse_args()

    if not os.path.exists(UPSTREAM_YAML):
        print(f"Error: {UPSTREAM_YAML} not found")
        sys.exit(1)

    with open(UPSTREAM_YAML) as f:
        config = yaml.safe_load(f)

    deps = config.get("dependencies", {})
    total_behind = 0

    print(f"\n{BOLD}TQBridge Upstream Dependency Check{RESET}\n")

    for name, dep in deps.items():
        print(f"{BOLD}{name}{RESET}")
        impact = dep.get("impact", "")
        if impact:
            print(f"  {DIM}Impact: {impact}{RESET}")
        behind = check_submodule(name, dep, fetch=args.fetch, verbose=args.verbose)
        total_behind += behind
        print()

    if total_behind == 0:
        print(f"{GREEN}All dependencies up to date.{RESET}\n")
    else:
        print(f"{YELLOW}{total_behind} total commits behind across dependencies.{RESET}")
        print(f"{DIM}Run 'git submodule update --remote <name>' to update.{RESET}\n")


if __name__ == "__main__":
    main()
