# Benchmark Analysis Prompt

You are analyzing one benchmark session from the MojoGP benchmark store.

## Session Context

- Session ID: `$SESSION_ID`
- Selector: `$SELECTOR`
- Benchmark filter: `$BENCHMARK_FILTER`
- GPU filter: `$GPU_NAME`
- Canonical DB: `$DB_PATH`

## Primary Inputs

Read these files first and treat them as the canonical analysis surface for this run:

1. `analysis_packet.json`: `$ANALYSIS_PACKET_PATH`
2. `source_manifest.json`: `$SOURCE_MANIFEST_PATH`

Use raw result JSON files only when `source_manifest.json` points to a specific file that needs verification.
Do not do broad repo searches unless the packet leaves a concrete ambiguity unresolved.

## What The Packet Contains

The packet contains these sections:

- `selected_session`
- `session_summary`
- `pairwise_compare`
- `scaling_steps`
- `fallback_audit`
- `raw_runs`
- `categorized_runs`
- `redflags`

## Required Workflow

1. Start with `session_summary` to understand coverage.
2. Use `pairwise_compare` for cross-framework fair rows.
3. Use `scaling_steps` to analyze scaling with `n`.
4. Use `fallback_audit` to identify MojoGP-only, unsupported, or fallback rows.
5. Use `redflags` as hypotheses to verify, not as final verdicts.
6. Use `raw_runs` when you need separate prediction mean/variance timing, CG telemetry, startup timing, or memory-phase fields.

## Hard Rules

1. Do not make cross-framework claims from rows labeled `mojogp_only`, `mojogp_only_scale`, or `unsupported_comparator`.
2. Treat `prediction_mean_time_s` and `prediction_variance_time_s` as separate measurements.
3. Do not rely only on total prediction time if the packet marks timing quality as combined or suspicious.
4. Discuss memory using phase-specific fields when available:
   - `training_peak_gpu_mb`
   - `training_delta_gpu_mb`
   - `prediction_peak_gpu_mb`
   - `prediction_delta_gpu_mb`
   - `exact_prediction_*`
   - `love_prediction_*`
5. Do not claim matrix-free memory is flat based only on absolute `gpu_max_mb`.
6. If `d`-scaling looks odd, inspect CG telemetry before inferring compute scaling.
7. Call out methodology limits explicitly, including fairness notes and timing-quality caveats.

## Required Analysis Dimensions

Cover all of the following when the packet contains the needed rows:

1. MojoGP vs GPyTorch materialized training and prediction.
2. MojoGP vs GPyTorch+KeOps matrix-free training and prediction.
3. MojoGP materialized vs matrix-free route comparisons.
4. Exact vs LOVE behavior within MojoGP.
5. Scaling with `n` for training, prediction, and memory.
6. Multi-output rows separately from single-output rows.
7. Preset sweep interpretation separately from fair scaling rows.
8. Benchmark methodology gaps and suspicious measurements.

## Deliverable Shape

Produce:

1. Executive summary.
2. Findings ordered by importance.
3. Cross-framework analysis.
4. Intra-MojoGP route analysis.
5. Memory analysis.
6. Methodology and measurement issues.
7. Concrete follow-up recommendations.

## Session Snapshot

- Run count in packet: `$RUN_COUNT`
- Pairwise fair comparisons: `$PAIRWISE_COUNT`
- Redflags emitted: `$REDFLAG_COUNT`
