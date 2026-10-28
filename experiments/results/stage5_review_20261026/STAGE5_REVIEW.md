# Stage 5 PAPER experiment review

Review date: 2026-10-26

## Scope and conclusion

This review audited the existing PAPER-FULL results and their event logs. It did
not launch any NPU search and did not modify the 450 historical result JSON
files.

All 450 records passed the result-level checks for success, correctness
acceptance, budget compliance, five-repeat final remeasurement, model identity,
RAG fingerprint, and result commit. All 450 event logs have exactly the same
number of `eval` events as the corresponding result's `npu_calls`. The main
result set is usable for writing, subject to the statistical and provenance
qualifications below.

## Main results

There are 50 kernels per configuration, each searched with seeds 11, 12, and
13. The primary statistic first takes the median speedup across the three
searches for each kernel, then calculates the geometric mean across kernels.

| Configuration | Geometric mean of per-kernel median speedup | Kernels with median speedup > 1 |
|---|---:|---:|
| B1 (`paper_b1`) | 1.5440 | 35/50 |
| B2 (`paper_b2`) | 1.5403 | 33/50 |
| FULL (`paper_full`) | 1.6531 | 39/50 |

Paired kernel-level comparisons of FULL:

| Comparison | Relative geometric-mean increase | Paired kernel bootstrap 95% CI | Wilcoxon two-sided p |
|---|---:|---:|---:|
| FULL vs B1 | 7.07% | 1.94% to 14.94% | 0.0060 |
| FULL vs B2 | 7.33% | 1.20% to 15.91% | 0.0255 |

The bootstrap resamples the 50 kernels (20,000 samples; seed 20261026). The
statistical unit is the kernel after reducing its three independent search
seeds by the median. Treat these as paired-kernel estimates, not as proof that
kernels are mutually independent.

Search cost:

| Configuration | Mean wall time per search | Mean LLM tokens per search | Total NPU calls |
|---|---:|---:|---:|
| B1 | 418.5 s | 37.6k | 1,950 |
| B2 | 481.7 s | 34.9k | 1,950 |
| FULL | 452.8 s | 47.2k | 1,935 |

The budget is an upper bound of 13 NPU calls per search. B1 and B2 each used
13 calls for all 150 searches. FULL used 13 calls in 141 searches, 12 calls in
7, 10 calls in 1, and 8 calls in 1 (1,935 total). All nine lower-call records
still passed their recorded correctness and final-result acceptance checks and
were within budget.

## Review of the nine lower-call FULL runs

Each row below is cross-checked against the JSONL event log. `perf_test_failed`
is the recorded event error label, not a claim that the overall run failed.
Task-log LLM retry counts are reported only where the log contains an actual
execution section for that exact seed; sections marked as reused are not used
to attribute retries.

| Kernel / seed | NPU calls | Preflight rejects | `perf_test_failed:unknown` events | Matching task-log LLM evidence |
|---|---:|---:|---:|---|
| `_act_quant_kernel` / 12 | 12 | 4 | 3 | Task section reused; attribution unavailable |
| `_convert_req_index_to_global_index_kernel` / 11 | 12 | 4 | 3 | Task section reused; attribution unavailable |
| `_dequantize_k_cache_fast_kernel` / 12 | 8 | 0 | 3 | Task section reused; attribution unavailable |
| `_log_softmax_kernel` / 13 | 12 | 4 | 0 | 0 connection retries, 0 terminal failures |
| `_pack_seq_kernel` / 12 | 10 | 0 | 4 | 21 connection retry messages; 3 terminal LLM failures |
| `assign_extend_cache_locs` / 11 | 12 | 4 | 2 | 0 connection retries, 0 terminal failures |
| `compute_identity_kernel` / 13 | 12 | 4 | 1 | 0 connection retries, 0 terminal failures |
| `l2norm_fwd_kernel1` / 12 | 12 | 4 | 2 | 0 connection retries, 0 terminal failures |
| `mean_kernel` / 13 | 12 | 0 | 0 | 11 connection retry messages; 1 terminal LLM failure |

The event logs show why some candidate attempts did not become measured
evaluations: several were rejected by preflight rules, while some performance
tests failed with the generic `unknown` label. The task log explicitly supports
LLM connection failures for `_pack_seq_kernel` and `mean_kernel`; it does not
provide matching execution history for the three rows marked reused. In
particular, the 8-call `_dequantize_k_cache_fast_kernel` record has no
run-specific task-log evidence that explains why its search stopped below the
cap. Its historical JSON records success and `budget_ok=true` but no
termination reason.

Recommendation: do not selectively replace these results. Report the protocol
as “up to 13 NPU evaluations per search” and preserve the original data. The
8- and 10-call rows have weaker evidence that the intended search budget was
fully exercised; if the paper makes a strict fixed-budget claim, first recover
their original run logs or rerun them only under a predeclared, symmetric
recovery policy applied to all affected records.

## Dataset provenance

The original result files have null dataset snapshot fields, so the historical
`dataset_sha256_matches_meta` flag is false. To close the provenance chain
without rewriting those records, this review created an external deterministic
snapshot manifest containing the 50 manifests, 150 input-file fingerprints,
source fingerprints, result/freeze commits, and RAG index fingerprint.

- Snapshot SHA-256: `ad1dc2686f89762d068563ed9abbeb902915e4f7f5ff7b20ebed7974ef7e97b3`
- The 450-row audit CSV now links each row to this snapshot digest.
- Source and experiment-input fingerprints were checked against the result
  commit and files in the working tree.
- The fixed RAG fingerprint is
  `3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397`.

This is an external provenance supplement; it does not retroactively change
the metadata inside the original results.

## Current host environment snapshot

The environment file records the observation timestamp and raw `npu-smi`
output. It is a snapshot of the host at review time and does not prove that
every setting was identical during the original experiment.

- Python 3.11.15; Linux aarch64.
- Two visible Ascend910_9382 devices; historical runs configured one device.
- Driver / `npu-smi`: 26.1.1; CANN OPP: 9.0.0.
- PyTorch 2.7.1; torch-npu 2.7.1.post4; `triton` 3.5.0; `triton-ascend`
  3.2.1.
- Historical results record five LLM workers and model ID
  `140aced515cc4236a6141f2b8a77107f`.

See `environment_snapshot.json` for full versions and raw device output.

## Frozen-commit regression tests

Tests were run in a temporary worktree at the exact result commit
`559f2f539c59886f155df92952141f210fa87c71`. The budget test needs the
gitignored fixture `experiments/dataset/samples_frozen_20261016.jsonl`; that
fixture was copied into the temporary worktree for the test and the worktree
was removed afterward.

| Test file | Result |
|---|---:|
| `experiments/tests/test_parent_latency.py` | 41 passed |
| `experiments/tests/test_filter_budget.py` | 13 passed |
| `experiments/tests/test_paper_protocol.py` | 7 passed |
| **Total** | **61 passed, 0 failed** |

The command outputs are preserved in `regression_tests/`. This result should be
described as passing with the ignored fixture supplied, not as passing in a
bare checkout without external test data.

## Reproduction

From the repository root:

```bash
python experiments/stage5_review.py
```

This verifies the historical JSONs, inputs, manifests, event-log evaluation
counts, RAG fingerprint, and regenerates the analysis artifacts. The regression
logs under `regression_tests/` contain the successful frozen-commit test runs.
