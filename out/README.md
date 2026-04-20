# out/

Experimental outputs from spike scripts and (later) benchmark runs. Most
files are git-ignored; only the OQ1 spike result JSON is tracked so the ADR
0001 decision is reproducible from the repository alone.

## Tracked files

- `oq1_results.json` — output of `scripts/spikes/oq1_codebook.py`. Populated
  *on Studio (M3 Ultra)* — not on the dev machine. Feeds into
  `docs/adr/0001-codebook-sharing.md`.

## Regenerating the OQ1 result

```bash
# on Studio
uv run python scripts/spikes/oq1_codebook.py \
    --mode both --seeds 5 --steps 2000 \
    --report out/oq1_results.json
```

Then update ADR 0001 with the numbers and commit both files together.
