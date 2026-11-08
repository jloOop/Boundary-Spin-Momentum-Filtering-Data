# Reproducing the centered saved-data results

Use Python3.12 and the small CPU environment below. The tested package versions are pinned separately from the preserved legacy environment. No networking, GPU, original research folder or private path is used by the reproduction commands after dependency installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-centered.txt
python Loaders/reproduce_centered.py --group main --output build/main
python Loaders/reproduce_centered.py --group diagnostics --output build/diagnostics
python Loaders/reproduce_centered.py --group controls --output build/controls
python Loaders/verify_centered.py
```

The new tracked package is approximately **179 MiB**, including about **177 MiB** in the centered data area. This exceeds the soft100 MiB target to retain all19 scientific/accounting columns for11 reference/refinement histories, plus their controls. Native step NPZ members use standard-library ZIP-LZMA compression; NumPy loads them through Python's `zipfile`/`lzma`. No selected numerical row or precision was sacrificed to reach a size cap.

Run these from the checkout root. Output directories are created automatically and ignored by Git. Entry points also locate inputs relative to their own checkout, independent of the caller's working directory. They reject writes into tracked input directories. All commands read committed saved data; they never import or execute generating GPU solvers.

| Group | What is actually supported |
|---|---|
| main | Native interval-aware click probabilities and moments, every-step survival integrals, main horizon statistics and empirical fit; original Simpson recipe and stated piecewise-linear check remain separate. Derived-data replots are identified separately from the archived displayed PNGs. |
| diagnostics | Native branch/spin and auxiliary diagnostics where records permit, archived Tables S2–S4 comparisons, and the exact four-run S5 refinement definitions. Missing scalar histories and complex spectral fields are explicit limits. |
| controls | Native upper/lower probability-channel checks and their late-time comparison; saved long-time norm/derived-profile checks. Parameter/bulk/chiral tables and current maps have archived-display status only. S12 has native-event replotting; S13 has derived-bin replotting. |

Each command writes numerical tables/figures and a JSON report. The verifier checks committed hashes, native schemas/support, interval and launch-norm consistency, analytic edge cases, declared numerical comparisons, local documentation links and protected-media content. Failure is nonzero status; missing-input items are reported as unavailable, not silently counted as reproduced. Numerical differences and tolerances are reported, without treating fit residuals as discretization errors.

The [single manifest](data/centered/provenance/manifest.json) links each figure/table to its input, adapter, original run/configuration/source and reproduction level. [PROJECT_SCOPE.md](PROJECT_SCOPE.md) lists source-version discrepancies. The [gallery](data/centered/README.md) links the actual tracked images and supporting tables.

## Supplementary table presentation

[Tables S1–S12](data/centered/tables/README.md) are generated from canonical CSVs by a small standard-library renderer. Existing high-precision records are reused; settings and four-run refinement summaries retain their numerical source. Tables S7–S9 explicitly contain **manuscript-reported rounded values**, with PDF page/hash provenance. Their presentation is not an independent numerical reconstruction. Implementation type and reproduction status are separate labels.

```bash
python Loaders/render_supplement_tables.py --check
python Loaders/render_supplement_tables.py
```

The first command checks that the committed catalog agrees with the CSVs, expected row counts, identifiers and declared display rounding, and writes nothing. The second regenerates only `data/centered/tables/README.md`; it never changes numerical data. Decimal tokens are formatted using round-to-nearest, ties-to-even. Display differences from the manuscript remain explicit. Run `verify_centered.py` for the combined presentation, link, identity and native-data checks.

The settings and four-run refinement summaries can be checked separately with `python -m Loaders.centered.supplement_tables_a --check --diagnostics-output build/diagnostics` after the diagnostics command. Omit `--diagnostics-output` to recompute only the four required refinement runs. The long-time summary uses `python -m Loaders.centered.supplement_tables_b --check`. The verifier also reports all 282 inspected S1–S6 printed-value comparisons, including the nine known disagreements, without changing the numerical results to match the manuscript.

The [gallery](data/centered/README.md) includes all thirteen supplementary figure presentations. Original archived exports and faithful manuscript figure extracts are identified separately, with extraction recipes in the manifest. Displaying an image does not imply that its curve-level inputs are bundled.

## Packaging versus routine reproduction

Tracked packaging/extraction scripts in `Loaders/centered/` accept an explicit local research root. They were used only to curate mapped sources, calculate source/distributed hashes, retain all selected rows and verify equality after export. Routine users do not need to run them or possess that original folder. They do not generate simulations. Any selected historical plotting/reduction recipes are provenance, separate from the portable adapters; machine paths and dependencies may have required documented transformations.

The compact native format preserves CSV column names, row ordering, accepted endpoints, stored masses, launch normalization, spin-event entries and advertised accounting columns. Only unneeded wall-clock timing is omitted from the main step exports. The manifest records actual dtypes, shapes, selected columns and lossless compression; no clipping, smoothing, downsampling or renormalization is applied to archived values. Small diagnostic arrays preserve their own clocks and validity masks. Loading uses `allow_pickle=False`.

CPU reproduction is not a fresh simulation, restart, new convergence calculation or experimental validation. Source snapshots in `Solvers/centered/` require production CUDA/CuPy dependencies and are not part of these CPU checks. Complete state/checkpoint collections, scalar native histories, layer histories and some monitor/spectral inputs are absent. Existing animations require neither regeneration nor download for this workflow.

## Verification record

The manifest records the tested environment and packaging equality results. `build/verify/` contains the reports generated by the verifier; these can be regenerated on a clean checkout. The protected media subtree is additionally checked by its initial Git tree/blob inventory and working-file hashes before commit and after push. Existing release identities and download URLs are compared read-only; no release is created or modified.
