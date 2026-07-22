# Public Codex A/B evaluation

This directory freezes a reproducible, evidence-scored comparison of four
read-only information surfaces:

1. native Codex file tools;
2. CodeGraph at commit
   `286e9ccc2dad45336d4fd67052930322054d64b5` with its default explore tool;
3. HLSGraph 0.2.0 at commit
   `7b26bfb07aa7c4d1a4705d3076cac684c3561e6f` with its narrow MCP tools; and
4. the HLSGraph 0.3.0 candidate with its single `explore` tool.

The model is fixed to `gpt-5.6-sol` with `medium` reasoning. There are 12
questions, four repetitions, and four arms: 192 cells. The runner randomizes
cell order with the checked-in seed. Its default behavior is a dry run; a model
is not contacted unless `--execute` is supplied.

## Truth and corpus policy

`corpus.lock.json` permits only Apache-2.0 inputs:

- the repository's synthetic `dataflow_gemm` fixture;
- AMD/Xilinx `using_stream_of_blocks` at the pinned 2024.2 commit;
- the pinned Vitis Libraries bitonic source and test; and
- the pinned pp4fpgas CORDIC source.

External files are downloaded from commit-qualified raw URLs and verified by
SHA-256. They are source-only. The local report-like files are explicitly
synthetic and must never be described as an AMD tool run. No proprietary report,
private source, model artifact, server setting, or external PDF belongs in this
suite.

Answers use `answer.schema.json`. Every material claim supplies a truth plane,
authority, stage, and project-relative line citation. `questions.jsonl` contains
frozen selectors rather than reference prose. The deterministic scorer checks
the cited bytes and truth plane directly, and re-verifies the locked corpus
before and after each score; no LLM judge controls the release gate.

## Reproduce

### Official execution platform and layout

The official profile is POSIX/Linux and is currently pinned to x86-64 Ubuntu
22.04 under WSL2. Native Windows is **dry-run only**. On Windows, an official
`prepare --execute` or `runner --execute` is a NO-GO and exits before creating,
changing, or deleting evaluation files. Do not treat a Windows dry run as an
official result.

Keep the complete execution runtime on the WSL ext4 filesystem, not under
`/mnt/c`, `/mnt/d`, another drvfs mount, a Windows junction, or a symlink into
one. In particular, all of the following must live on ext4:

- `work_root` and `runs_root`;
- the clean v0.2 and v0.3 virtual environments;
- the CodeGraph checkout, its Linux Node runtime, and built entrypoint;
- the official Linux Codex CLI `0.144.0`, its `codex-linux-sandbox` helper, and
  `bwrap`.

Preparation records the WSL distribution/kernel/architecture and the exact
lexical Python, Node, Codex, sandbox-helper, `bwrap`, and CodeGraph entrypoint
paths, versions, and SHA-256 identities. A venv `bin/python` symlink remains the
selected executable path rather than being rewritten to its system target; the
symlink must resolve to an existing regular file on ext4. Execution fails
closed if any locked byte or identity changes. A
typical ext4-only layout is:

```bash
export HLSGRAPH_REPO="$HOME/src/hlsgraph-public"
export EVAL_ROOT="/var/tmp/hlsgraph-agent-ab-$USER"
export RUNTIME_ROOT="$EVAL_ROOT/runtime"
export WORK_ROOT="$EVAL_ROOT/work"
export RUNS_ROOT="$EVAL_ROOT/runs"
export RESULTS_ROOT="$EVAL_ROOT/results"
export V02_PYTHON="$RUNTIME_ROOT/venvs/v02/bin/python"
export V03_PYTHON="$RUNTIME_ROOT/venvs/v03/bin/python"
export V02_REPO="$RUNTIME_ROOT/hlsgraph-v02-src"
export CODEGRAPH_REPO="$RUNTIME_ROOT/codegraph"
export CODEGRAPH_JS="$CODEGRAPH_REPO/dist/bin/codegraph.js"
export CODEX_BIN="$RUNTIME_ROOT/codex/codex"
mkdir -p -m 700 "$EVAL_ROOT" "$RUNTIME_ROOT" "$RESULTS_ROOT"

# Every line must report ext4 before an official preparation or run.
findmnt -T "$EVAL_ROOT"
findmnt -T "$V02_PYTHON"
findmnt -T "$V03_PYTHON"
findmnt -T "$CODEGRAPH_JS"
findmnt -T "$CODEX_BIN"
findmnt -T "$(command -v bwrap)"
```

Use Python 3.10 or newer. Create the two clean environments outside the public
checkout, one containing the published 0.2 wheel and one containing the
candidate 0.3 wheel. Keep clean source checkouts for both wheels; `V02_REPO`
must be exactly commit `7b26bfb07aa7c4d1a4705d3076cac684c3561e6f`, while
`HLSGRAPH_REPO` is the clean v0.3 candidate checkout. Record both wheel hashes
with the result bundle. Editable
or source-directory installs are rejected: preparation compares every installed
package and distribution-metadata byte against the supplied wheel, records both
source revisions before and after indexing, and proves that every package byte
and path in each source checkout is identical to its wheel payload. This binds
both evaluated artifacts to reviewable source commits.

Install the official Linux Codex CLI `0.144.0` release and its sandbox helper
under `$RUNTIME_ROOT/codex`. Use a **new, dedicated** `CODEX_HOME` that is
outside and disjoint from `WORK_ROOT`. Authenticate it interactively; never copy
the default `~/.codex`, an auth file, or any other home directory into it. API
key/token environment authentication is forbidden for this evaluation:

```bash
export CODEX_HOME="$HOME/.codex-hlsgraph-agent-ab"
unset OPENAI_API_KEY CODEX_API_KEY CODEX_ACCESS_TOKEN
mkdir -p -m 700 "$CODEX_HOME"
CODEX_HOME="$CODEX_HOME" "$CODEX_BIN" login
"$CODEX_BIN" --version                 # must be: codex-cli 0.144.0
```

The runner passes only the dedicated `CODEX_HOME`; it does not fall back to the
default home or forward the forbidden authentication variables.

Build CodeGraph from the frozen commit on ext4. The resulting
`dist/bin/codegraph.js` must match the SHA-256 frozen in `manifest.json`; a
matching Git commit with different entrypoint bytes is rejected. Disable its
daemon, telemetry, and update checks for preparation, indexing, and every model
cell. These four values are part of the locked execution contract and scoring
rejects a missing or changed value:

```bash
export CODEGRAPH_NO_DAEMON=1
export CODEGRAPH_TELEMETRY=0
export DO_NOT_TRACK=1
export CODEGRAPH_NO_UPDATE_CHECK=1

git clone https://github.com/colbymchenry/codegraph.git "$CODEGRAPH_REPO"
git -C "$CODEGRAPH_REPO" checkout 286e9ccc2dad45336d4fd67052930322054d64b5
npm --prefix "$CODEGRAPH_REPO" ci
npm --prefix "$CODEGRAPH_REPO" run build
```

Materialize the isolated workspaces. Network access is confined to this explicit
dependency/corpus preparation phase and the pinned public inputs; model cells
have no network permission:

```bash
cd "$HLSGRAPH_REPO"
python3 -m eval.agent_ab.setup_corpus \
  --repo-root "$HLSGRAPH_REPO" --output "$WORK_ROOT" --fetch
```

Inspect the preparation plan first, then opt in to local indexing. Indexing does
not invoke Vitis, Vivado, simulation, implementation, a remote host, or training:

```bash
export V02_WHEEL="$HLSGRAPH_REPO/dist/hlsgraph-0.2.0-py3-none-any.whl"
export V03_WHEEL="$HLSGRAPH_REPO/dist/hlsgraph-0.3.0-py3-none-any.whl"
export CODEGRAPH_COMMAND="$(command -v node) $CODEGRAPH_JS"

python3 -m eval.agent_ab.prepare \
  --work-root "$WORK_ROOT" \
  --v02-python "$V02_PYTHON" \
  --v03-python "$V03_PYTHON" \
  --codegraph-command "$CODEGRAPH_COMMAND" \
  --codegraph-repo "$CODEGRAPH_REPO" \
  --v02-repo "$V02_REPO" \
  --v03-repo "$HLSGRAPH_REPO" \
  --codex-command "$CODEX_BIN" \
  --v02-wheel "$V02_WHEEL" --v03-wheel "$V03_WHEEL"

python3 -m eval.agent_ab.prepare \
  --work-root "$WORK_ROOT" \
  --v02-python "$V02_PYTHON" \
  --v03-python "$V03_PYTHON" \
  --codegraph-command "$CODEGRAPH_COMMAND" \
  --codegraph-repo "$CODEGRAPH_REPO" \
  --v02-repo "$V02_REPO" \
  --v03-repo "$HLSGRAPH_REPO" \
  --codex-command "$CODEX_BIN" \
  --v02-wheel "$V02_WHEEL" --v03-wheel "$V03_WHEEL" --execute
```

The release profile requires libclang in both environments. `--degraded` exists
only for parser diagnostics; an environment lock produced with that flag is
marked non-official and cannot support a performance claim.
Official execution requires both HLSGraph checkouts to be clean and records their
Git revisions, the exact CodeGraph entrypoint path/hash, Node/Python binary identities,
the Codex CLI and Linux sandbox identities, and both verified wheel payload
identities.
Preparation also installs the complete built-in knowledge catalog, including
its audited executable-rule and citation-only classifications, into each v0.3
evaluation ledger. The generated manifests declare AMD
Vitis/Vivado 2024.2 only as applicability context; this is not executed-tool
evidence.
Cold-start accounting is phase-explicit: CodeGraph and HLSGraph v0.2 record an
`index` phase, while HLSGraph v0.3 records both `index` and `knowledge_sync`.
The aggregate cold-start duration and command identity are derived from those
ordered phases; omitting the v0.3 knowledge sync invalidates the lock.

Collect and score the deterministic top-8 retrieval gate before any model runs:

```bash
"$V03_PYTHON" -m eval.agent_ab.static_eval collect \
  --work-root "$WORK_ROOT" \
  --output "$RESULTS_ROOT/static-results.json" --execute
"$V03_PYTHON" -m eval.agent_ab.static_eval score \
  "$RESULTS_ROOT/static-results.json" \
  --work-root "$WORK_ROOT" \
  --output "$RESULTS_ROOT/static-report.json"
```

Review the 192-cell dry-run plan before the expensive step:

```bash
python3 -m eval.agent_ab.runner \
  --work-root "$WORK_ROOT" --runs-root "$RUNS_ROOT" \
  --codex-command "$CODEX_BIN" \
  --v02-python "$V02_PYTHON" --v03-python "$V03_PYTHON" \
  --codegraph-command "$CODEGRAPH_COMMAND" > "$EVAL_ROOT/eval-plan.json"

python3 -m eval.agent_ab.runner \
  --work-root "$WORK_ROOT" --runs-root "$RUNS_ROOT" \
  --codex-command "$CODEX_BIN" \
  --v02-python "$V02_PYTHON" --v03-python "$V03_PYTHON" \
  --codegraph-command "$CODEGRAPH_COMMAND" --execute
```

Every cell explicitly sets approval to `never`, uses the read-only sandbox, and
disables browser, computer-use, app, delegation, plugin, hook, and workspace-
dependency features. Only the built-in read-only shell/file surface and the
single arm-specific MCP server remain available.
For every graph arm, the first tool call must be that arm's exact treatment MCP
tool and at least one such call must appear. A failed or unfinished first MCP
attempt is retained explicitly as `failed` or `incomplete`; it is not silently
reclassified as a native/file-only run.

The OS sandbox is the primary confidentiality boundary. It denies the public
repository, `CODEX_HOME`, every discovered `/mnt/<drive>` drvfs root, sensitive
home roots, and an external private-looking canary root. Within `work_root`, the
locked `explicit_sibling_directory_deny_v1` policy denies the other three arm
directories, the other three corpus directories under the current arm, and the
sealed control directories; only the current workspace remains readable. The
runner requires an exact top-level and per-arm directory inventory before and
after use, so an unknown file or directory is a NO-GO. The two known readable
root-level metadata files contain no questions or gold evidence;
`environment.lock.json` is additionally byte-bound into the run set.

This directory-deny form is deliberate. Codex CLI 0.144.0's Linux `bwrap`
backend cannot reopen a readable child after its parent has already been
mounted as unreadable; an apparent `work_root=deny` plus child `read` policy
therefore makes the current workspace return `ENOENT`. Explicit sibling
directory denies provide the enforceable equivalent without relying on a
prompt or trace check. Trace inspection remains a second audit layer, not a
substitute for the OS boundary.

Before the first model call, the runner executes fail-closed permission
canaries. They prove that the current workspace is readable while both a
same-arm sibling and an other-arm sibling, the boundary-control directory, the
exact `runs_root`, the public gold repository, synthetic files in the dedicated
`CODEX_HOME` and user home, an external/private-like sibling, and a fresh
sentinel on every drvfs mount are unreadable; a local socket canary also proves
network denial. Any unexpected allow, any failed allow, or an untestable mount
aborts the batch.

Before execution, the runner verifies that `runs_root` is an unlinked ext4
directory disjoint from every sandbox deny root. Every cell's OS permission
profile contains an explicit deny for that exact root, and both the canary
receipt and scoring-time command validation bind it. The runner then writes one
immutable `run-set.json` for the exact 192-cell matrix. It binds a random batch
ID, each question/arm/repetition,
the exact `runs_root`, the fixed 900-second timeout, the Codex executable and
arm-specific MCP executable/entrypoint argv, prompt hash, candidate/index
workspace identity, and a distinct per-cell trace challenge. Those command
paths and arguments are checked against the path and byte identities in
`environment.lock.json`. The answer must echo that challenge in its final
`uncertainties` entry. Consequently, renaming two run directories or swapping
their traces cannot turn one cell into another. Scoring also requires exactly
one unique Codex thread ID and unique raw JSONL hash per successful cell.
The body-free private-retrieval access log is the only excluded mutable index
file; the runner safely removes it before every cell so it cannot carry context
between arms or repetitions. Every other workspace byte, including unexpected
files, is covered by the frozen workspace identity.
The CodeGraph JS bytes and both installed HLSGraph payloads are rechecked after
every model cell and again at batch completion. Each successful score also
requires usage from a terminal Codex completion event with all token counters
present and `total_tokens > 0`; missing, synthesized, or zero terminal usage is
a failed cell.

Raw traces stay under the ignored `runs/` directory. Score and bootstrap them:

```bash
python3 -m eval.agent_ab.score \
  --runs-root "$RUNS_ROOT" --work-root "$WORK_ROOT" \
  --output "$RESULTS_ROOT/scores.jsonl"
python3 -m eval.agent_ab.bootstrap "$RESULTS_ROOT/scores.jsonl" \
  --static-report "$RESULTS_ROOT/static-report.json" \
  --runs-root "$RUNS_ROOT" \
  --work-root "$WORK_ROOT" \
  --v03-python "$V03_PYTHON" \
  --output "$RESULTS_ROOT/bootstrap.json"
```

Bootstrap does not trust either supplied score artifact. It deterministically
re-scores all 192 ignored raw traces and requires a byte-for-byte match with
`scores.jsonl`. It then starts the pinned candidate interpreter in isolated
mode, re-collects and re-scores all static queries against the frozen v0.3
indexes, and requires a byte-for-byte match with `static-report.json`. NaN,
infinity, out-of-range quality ratios, negative counts/timings, stale snapshot
or index identities, duplicate traces, and a dirty/untracked/linked candidate
checkout fail closed before any performance claim is evaluated.

Before publishing any trace, sanitize it into a separate directory and audit the
sanitized output. The sanitizer removes command/MCP request and response
payloads as well as credentials and absolute paths. Never publish the ignored
raw directory:

```bash
python3 -m eval.agent_ab.sanitize raw.jsonl public.jsonl \
  --workspace '<absolute-workspace-root>'
python3 -m eval.agent_ab.audit public.jsonl
```

The bootstrap is paired by question and repetition, stratified by question, and
uses 10,000 seeded resamples. Positive deltas always favor HLSGraph 0.3. The
report's `performance_advantage_supported` field is the only permission to claim
an advantage; a false value must be reported as a technical preview.

## Development checks

These commands do not start any model calls:

```bash
python3 -m eval.agent_ab.audit
python3 -m eval.agent_ab.runner --arm native --question dg-architecture-flow
python3 -m pytest tests/test_agent_eval.py -q
```

The last three commands are safe as dry-run/development checks on either WSL2
or Windows. Windows must not add `--execute`.
