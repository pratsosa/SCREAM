#!/usr/bin/env python3
"""
anonymize.py — Run this from the anon branch to scrub all identifying terms.

Actions:
  1. Read anonymize_patterns.txt and apply all regex substitutions to tracked
     text files (.py, .md, .yaml, .yml, .sh, .toml, .txt).
  2. git rm all tracked .ipynb files and append *.ipynb to .gitignore.
  3. Delete anonymize_patterns.txt from disk.

After this script finishes, stage and commit everything:
    git add -A && git commit -m "Anonymized"
"""

import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
PATTERNS_FILE = REPO_ROOT / "anonymize_patterns.txt"
REPLACEMENT = "XXXX"

TARGET_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".sh", ".toml", ".txt"}

SKIP_DIRS = {".git", "data", "wandb", "__pycache__"}
SKIP_SUFFIX_PARTS = {".egg-info"}  # any dir whose name ends with .egg-info


def is_skipped(path: Path) -> bool:
    """Return True if path is under a directory that should be skipped."""
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
        if any(part.endswith(s) for s in SKIP_SUFFIX_PARTS):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 1: Load patterns
# ---------------------------------------------------------------------------

if not PATTERNS_FILE.exists():
    print(f"ERROR: {PATTERNS_FILE} not found. Must be run from the anon branch with the file present.")
    sys.exit(1)

raw_lines = PATTERNS_FILE.read_text().splitlines()
patterns = []
for line in raw_lines:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    patterns.append(re.compile(line, re.IGNORECASE))

print(f"Loaded {len(patterns)} pattern(s) from {PATTERNS_FILE.name}")

# ---------------------------------------------------------------------------
# Step 2: Walk repo and apply substitutions
# ---------------------------------------------------------------------------

replaced_files = []

for path in sorted(REPO_ROOT.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(REPO_ROOT)
    if is_skipped(rel):
        continue
    if path.suffix not in TARGET_EXTENSIONS:
        continue
    # Don't process this script itself (it contains no identifying terms anyway,
    # but we skip it to avoid confusing output).
    if path == Path(__file__).resolve():
        continue

    original = path.read_text(errors="replace")
    scrubbed = original
    for pat in patterns:
        scrubbed = pat.sub(REPLACEMENT, scrubbed)

    if scrubbed != original:
        path.write_text(scrubbed)
        replaced_files.append(rel)

if replaced_files:
    print(f"\nScrubbed {len(replaced_files)} file(s):")
    for f in replaced_files:
        print(f"  {f}")
else:
    print("\nNo files required scrubbing.")

# ---------------------------------------------------------------------------
# Step 3: Remove tracked .ipynb files
# ---------------------------------------------------------------------------

result = subprocess.run(
    ["git", "ls-files", "*.ipynb"],
    capture_output=True, text=True, cwd=REPO_ROOT
)
# git ls-files with a glob only matches top-level; use --recurse-submodules
# alternative: list all tracked files and filter manually
result2 = subprocess.run(
    ["git", "ls-files"],
    capture_output=True, text=True, cwd=REPO_ROOT
)
tracked_ipynb = [f for f in result2.stdout.splitlines() if f.endswith(".ipynb")]

if tracked_ipynb:
    print(f"\nRemoving {len(tracked_ipynb)} tracked notebook(s):")
    for nb in tracked_ipynb:
        print(f"  {nb}")
    subprocess.run(["git", "rm", "--"] + tracked_ipynb, cwd=REPO_ROOT, check=True)
else:
    print("\nNo tracked .ipynb files to remove.")

# Append *.ipynb to .gitignore if not already there
gitignore = REPO_ROOT / ".gitignore"
content = gitignore.read_text() if gitignore.exists() else ""
if "*.ipynb" not in content:
    with gitignore.open("a") as f:
        f.write("\n# Notebooks excluded from anon branch\n*.ipynb\n")
    print("Appended *.ipynb to .gitignore")

# ---------------------------------------------------------------------------
# Step 4: Delete the patterns file
# ---------------------------------------------------------------------------

PATTERNS_FILE.unlink()
print(f"\nDeleted {PATTERNS_FILE.name}")

print("\nDone. Now run:")
print("    git add -A && git commit -m 'Anonymized'")
