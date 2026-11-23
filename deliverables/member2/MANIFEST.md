# Package manifest

| directory / file | files | size |
|---|---|---|
| `README.md` | 1 | 0.0 MiB |
| `SUMMARY.md` | 1 | 0.0 MiB |
| `baseline_gate` | 3 | 0.0 MiB |
| `code` | 23 | 0.1 MiB |
| `infos` | 5 | 109.0 MiB |
| `official_eval` | 10 | 0.0 MiB |
| `process.txt` | 1 | 0.1 MiB |
| `provenance` | 11 | 3.8 MiB |
| `results_summary.json` | 1 | 0.0 MiB |
| `runs` | 30 | 521.8 MiB |
| `schedules` | 3 | 0.1 MiB |
| `scripts` | 3 | 0.0 MiB |
| `verification` | 6 | 0.0 MiB |

Verify: `cd <this directory> && sha256sum -c SHA256SUMS`

Built from the StreamPETR project by `python -m outage.collect_deliverables`; paths mentioned in `README.md` (a copy of the project README) are relative to that project root (`../StreamPETR/`), not to this package.

Deliberately EXCLUDED (invalid or diagnostic, shim stack): mini_*, pilot_a/b/d05/d10/d20, pilot_notf32, pilot_shimfp16, val_a, val_d05, val_notf32, ref_test raw outputs (only comparison.json is included, labelled SHIM_STACK under baseline_gate/), isolation_{mini,pilot}.json (shim stack), verification_{mini,pilot}.json (shim stack).

Not included (too large / external): raw nuScenes data (`~/storage/nuscenes*`), the StreamPETR source tree (`streampetr_src/`, pinned commit in SUMMARY), the checkpoint (`ckpts/`, sha256 in run.json), the flash-attn source (`fa_src`, commit in provenance).
