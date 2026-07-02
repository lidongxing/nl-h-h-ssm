# docs/

Per-file description stubs for reproducibility review.

| Path | Purpose |
|------|---------|
| `descriptions/` | One `.md` per tracked repo file — **unique** text (not copy-pasted); mirrors source tree |
| `../REPO_MANIFEST.json` | Same content as JSON (`path` → `description`) for scripts |

> **Note:** GitHub’s file list column **“Last commit message”** shows the git commit that last *modified* each file.  
> Source code and JSON logs may still share an older commit message.  
> Open **`docs/descriptions/<path>.md`** or **`REPO_MANIFEST.json`** for the dedicated per-file explanation.

Regenerate after adding files:

```bash
python scripts/generate_repo_manifest.py
```
