# System Benchmarks

`tests/system_benchmarks/` is the benchmark-only surface for MojoGP.

Rules:

1. benchmark suites in this directory must run through the subprocess harness
2. benchmark suites must persist sessions/cases/comparisons through `tests/benchmarks/`
3. benchmark suites may use queryable SQLite history as an empirical reference
4. benchmark suites must not be the only proof of mathematical correctness
5. small-n oracle tests, dense-reference checks, and API-only workflow tests belong in `tests/unit/`, `tests/integration/`, or `tests/system/`

Directory contents should be limited to:

1. harness benchmark entry files
2. benchmark child runner modules such as `run_*_case.py`
3. benchmark-specific helper code that is not correctness-test infrastructure

Do not add new in-process benchmark suites here.
