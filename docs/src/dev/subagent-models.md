# Subagent Model Recommendations

When running Claude Code with parallel or serial subagents — for CI scripts, feature work,
algorithm research, or documentation — the right model choice avoids paying for power you don't
need and avoids hitting limits on the tasks that do need deep context.

## Model tiers

### Haiku — boilerplate, CI, workflow files

Use Haiku for tasks where the output is **well-defined and mechanical**:

- GitHub Actions / CI workflow YAML
- Formatting / linting configuration (`.runic.toml`, etc.)
- Changelog and release-note generation from commit messages
- Dependency bumps and `Project.toml` edits
- Simple test stubs from function signatures
- Scaffold generation (new file from a template)

Haiku is fast and cheap. If the task fits in a tight, concrete prompt with a clear acceptance
criterion, Haiku will produce a correct result faster than Sonnet and at a fraction of the cost.

### Sonnet — docs, well-scoped features, refactors

Use Sonnet for tasks that need **sustained reasoning over a few files** but have a clear scope:

- Writing or updating documentation pages (like this site)
- Implementing a well-specified feature (e.g. the `prfft`/`pirfft` real-input variant once the
  interface is defined)
- Refactoring a module with a known target structure
- Writing benchmark scripts and plotting utilities
- Code review at "medium" depth (correctness + simplification)
- Translating a known algorithm from pseudocode or another language to Julia

Sonnet handles multi-file coherence well and can hold enough context to keep a feature's API,
tests, and docs consistent in a single session.

### Opus — hard perf/algorithm work, deep context

Use Opus for tasks where **correctness or performance depends on understanding the full system**:

- Designing or debugging a new FFT variant (e.g. a recursive multi-level four-step)
- Diagnosing a performance regression that requires reading profiler output + LLVM IR + benchmark
  history simultaneously
- Architecture decisions that touch multiple abstraction layers (plan dispatch, codelet generation,
  SIMD layout)
- Security or correctness reviews that require tracking invariants across many files
- Any task where you'd say "I need it to read REPORT.md, three source files, and the benchmark
  output before answering"

Opus is the right choice when the cost of a wrong answer (regression, architectural dead end) is
high. For PureFFT specifically: most of the remaining gap to FFTW is in orchestration architecture
(recursive multi-level decomposition, codelet scheduling), which is exactly the kind of deep,
multi-file reasoning Opus handles better.

## Practical rules for subagents on this project

**Subagents start cold.** They have no memory of prior sessions. A subagent that opens
`REPORT.md` without being told to will miss the history of what was tried and why. Always include
in the prompt:

- The relevant excerpt from `REPORT.md` or `bench/rustfft_compare/README.md`
- The specific file paths and line numbers to read
- The known negative findings (so it doesn't retry a dead end)

**Prefer parallel subagents for disjoint scopes.** Tasks that touch different files and have no
shared state are safe to run in parallel. Examples:

- Haiku writing a GitHub Actions workflow while Sonnet writes a new docs page
- Two Sonnet agents implementing `prfft` and `pirfft` if their implementations are file-separated
- A Haiku formatting pass running while Sonnet writes tests

Tasks that share state (the same source file, a plan type used by multiple variants) must be
serialized or carefully coordinated.

**Match brief depth to model tier.** For Haiku, a tight spec is correct and sufficient ("add a
CI job that runs `julia --project=bench bench/compare.jl` on push to main"). For Opus, invest in
a rich brief — include context, constraints, and known-wrong directions, because Opus will explore
deeply and a shallow brief wastes its capacity on rediscovering what you already know.

## Quick reference

| Task type | Model | Brief depth |
|---|---|---|
| CI/CD, YAML, config | Haiku | Tight spec |
| Docs, guides, READMEs | Sonnet | Moderate context |
| New feature (rfft, new variant) | Sonnet | Spec + API contract |
| Refactor, code review | Sonnet | Scope + acceptance test |
| Perf diagnosis, algorithm design | Opus | Full history + constraints |
| Architecture decisions | Opus | Full history + known dead ends |
