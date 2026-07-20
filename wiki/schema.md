# Wiki Schema

Following [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Layers

1. **Raw sources** (`raw/`)
   - `catalog.tsv`: index of ingested sources with sha256
   - `source-notes/<id>.md`: one note per raw source (research reports, model configs, benchmark data, etc.)
   - Immutable — never rewrite; append new notes if source updates

2. **Wiki** (`wiki/`)
   - `index.md`: master navigation catalog
   - `log.md`: append-only chronological record (mirrors CHANGELOG but wiki-anchored)
   - `schema.md`: this file
   - Topic pages organized by directory:
     - `models/` — one page per LLM (SmolLM2, Qwen, Midm...)
     - `hardware/` — T527 NPU, driver stack
     - `pipeline/` — pegasus stages: import, quantize, export
     - `techniques/` — SmoothQuant, patch scripts, sliding-window decode
     - `issues/` — bugs found & workarounds (Acuity axis, slice, saturation)
     - `decisions/` — architecture/recipe choices with rationale
     - `results/` — benchmark tables per experiment

3. **Schema** (this doc)

## Operations

- **Ingest**: new experiment result → source-note + touch relevant wiki pages + log entry + catalog line
- **Query**: read `index.md` → follow cross-refs
- **Lint**: no orphaned pages, all `[[link]]` refs resolve, log entries in order

## Cross-reference convention

Use markdown links: `[title](../pipeline/quantization.md)` from wiki pages.
For raw sources: `[source-note](../../raw/source-notes/src-<id>.md)`.

## Page template

```markdown
# Page Title

**Status**: draft | verified | stale
**Last updated**: YYYY-MM-DD
**Related**: [[other-page]], [[another-page]]

## Summary
One paragraph.

## Details
...

## Sources
- [source-note-id](path)
- external URLs
```

## Update rules

- Every substantive commit MUST touch at least one wiki page and add a `log.md` entry
- Bug findings → new `issues/` page + link from affected `pipeline/` or `models/` pages
- Benchmark results → new `results/` page + update summary in `models/<model>.md`
- Never delete pages; mark `Status: stale` and add successor link
