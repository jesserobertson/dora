# Dora → JAX/BlackJAX Migration Plan

## 0. Key finding that reframes this migration

`dora/active_sampling/gp_sampler.py` does **not** use `dora/regressors/gp/`.
It imports `revrand.legacygp` (an external, unpublished NICTA package fetched
via `pip install git+...`) and calls `gp.learn`, `gp.condition`, `gp.query`,
`gp.mean`, `gp.variance`, `gp.criterions.stacked_negative_log_marginal_likelihood`.

`dora/regressors/gp/` — `kernel.py`, `linalg.py`, `predict.py`, `train.py`,
`types.py` — is a self-contained, currently **orphaned** parallel GP
implementation with no in-repo caller. Confirmed: nothing outside
`regressors/gp` imports it, and the only test (`tests/test_sampler.py`)
exercises the `revrand`-backed path exclusively.

This changes what "port to JAX" means: it is not a mechanical translation of
live code, it's **promoting `regressors/gp` from dead code to the sampler's
real backend**, ported to JAX, replacing `revrand` entirely. Two
consequences that drive the plan below:

- The external `revrand` git dependency disappears — a real win (one
  unmaintained external package removed).
- The APIs don't line up today. `revrand.legacygp.learn()` is called with
  `opt_criterion=stacked_negative_log_marginal_likelihood` — a
  **multi-task-stacked** NLML — while `regressors/gp/train.py` only
  implements a single-task `'logMarg'`/`'crossVal'` objective. The port has
  to add a stacked/multi-task NLML variant; it is not a drop-in swap.

---

## 1. Recommended stack

| Layer | Choice | Role |
|---|---|---|
| Array math / autodiff | **JAX** (`jax.numpy`, `grad`, `vmap`, `jit`) | Replaces numpy in kernel/linalg/predict; free exact gradients |
| MLE hyperparameter training | **optax** (Adam warm-start, then L-BFGS) | Replaces `nlopt` for the default point-estimate training mode |
| Fully-Bayesian hyperparameters (opt-in) | **BlackJAX NUTS** | Posterior over kernel hyperparameters instead of a point estimate |
| GP model/kernel layer | **Hand-rolled JAX port of `dora/regressors/gp`** | See tradeoff below — not GPJax/TinyGP |
| Delaunay sampler | **Unchanged, scipy-based** | Not a Bayesian model, no JAX/BlackJAX involvement |
| Server | **FastAPI + pydantic** (replacing Flask) | Not JAX-related, but part of the rewrite (Phase 4) |

**BlackJAX's actual role:** BlackJAX is an inference-algorithm library
(NUTS, HMC, SMC, VI, window adaptation) over a user-supplied log-density
function on a pytree of parameters. It has no kernels, no GP posterior
machinery, no acquisition functions. It plugs in specifically as an
opt-in upgrade from point-estimate (MLE) hyperparameters to a full
posterior — it is not the thing that makes "Dora work in JAX" by itself.

**Why hand-roll instead of GPJax/TinyGP:** Dora's kernel zoo is
non-standard — `non_stationary`/`tree`/`tree1D` implement an
input-dependent-lengthscale kernel, and `nonstat_rr` recursively calls
`train.condition`/`predict.query` *inside* a kernel evaluation (a
GP-regressed lengthscale field). GPJax/TinyGP's kernel abstractions assume
static, non-recursive covariance functions — fitting `nonstat_rr` in means
fighting the library rather than using it, while the "normal" kernels
(gaussian/laplace/sin/matern3on2/chisquare) are trivial `cdist`→`jnp`
swaps anyway. Adopting a GP library buys little here and costs an
abstraction fight plus a heavier dependency. Tradeoff accepted: more
Cholesky/predict plumbing ourselves, in exchange for 100% behavioral
fidelity and one migration instead of two.

**Delaunay sampler:** left untouched. It's not a Bayesian/probabilistic
model — no gradients, no inference — so JAX/BlackJAX add nothing, and
`scipy.spatial.Delaunay`'s incremental triangulation has no JAX equivalent
anyway.

---

## 2. Module-by-module mapping

| Current file | Disposition | Notes |
|---|---|---|
| `regressors/gp/kernel.py` (`gaussian`, `laplace`, `sin`, `matern3on2`, `chisquare`, `mt_weights`) | **Port mechanically** to `jax.numpy` | `cdist` → manual pairwise-sqdist broadcasting (JAX has no `cdist`) |
| `kernel.py` (`non_stationary`/`tree`/`tree1D`/`nonstat_rr`) | **Defer / redesign** | Recursive kernel-calls-GP pattern doesn't jit cleanly; nothing in `active_sampling` currently uses these, so they're not required for parity |
| `kernel.py` (`compose`/`named_target`/`auto_range`/`describer`/`Printer`) | **Drop**, replace with typed pytree kernel objects | String-keyed `globals()` lookup + closure-based hyperparameter iteration isn't pytree-friendly (see §4) |
| `regressors/gp/linalg.py` (`jitchol`, `cholesky`, `choleskyjitter`) | **Port mechanically** | `scipy.linalg.cholesky` → `jax.scipy.linalg.cholesky`; the jitter retry-loop becomes `jax.lax.while_loop`/bounded `fori_loop` to stay jit-compatible |
| `regressors/gp/predict.py` | **Port mechanically** | `scipy.linalg.solve_triangular` → `jax.scipy.linalg.solve_triangular` |
| `regressors/gp/train.py` | **Replace optimizer, port objective math, add stacked NLML** | `nlopt` → `optax`/`jax.scipy.optimize.minimize` using `jax.grad`; add the multi-task stacked NLML `gp_sampler.py` needs (doesn't exist today) |
| `train.py` (`make_folds`) | **Keep as scipy/sklearn**, called outside jit | One-off preprocessing, not hot-path; no benefit to porting |
| `regressors/gp/types.py` | **Redesign as JAX pytree dataclasses** | See §4 |
| `active_sampling/base_sampler.py` | **Keep as plain Python/numpy** | Mutable orchestration shell, deliberately outside JAX — see §3 |
| `active_sampling/gp_sampler.py` | **Redesign internals, keep public API** | Rewire off `revrand.legacygp` onto the new JAX `regressors/gp`; `pick`/`update`/`predict` signatures unchanged |
| `active_sampling/delaunay_sampler.py` | **Keep as-is** | scipy-based, not a JAX candidate |
| `server/server.py`, `response.py` | **Replace with FastAPI + pydantic** | Phase 4, not JAX-related |
| `revrand.legacygp` (external dep) | **Drop entirely** | Superseded by the promoted, ported `regressors/gp` |
| `nlopt` | **Drop** | Fully replaced by `optax` |
| `sklearn` (clustering only) | **Keep** | `make_folds` stays scipy-side |

---

## 3. Orchestration: functional core, imperative shell

Keep `Sampler`/`GaussianProcess` as ordinary mutable Python objects
(`ArrayBuffer`, `pending_results: dict[uid -> index]`, `uuid.uuid4().hex`
job IDs) exactly as today. Push only the numerically heavy work — train,
condition, predict, acquisition scoring — into pure, `@jax.jit`-decorated
functions operating on plain arrays or small immutable pytree structs.

`pick()`/`update()` stay regular Python methods doing buffer/dict
bookkeeping in numpy, calling out to jitted pure functions:
`gp_core.train(X, y, kerneldef, ...) -> hyperparams`,
`gp_core.condition(...) -> RegressionState`,
`gp_core.predict(state, Xq) -> (mean, var)`,
`gp_core.acquire(state, Xq) -> scores`.

**Why not go fully functional** (threading an immutable sampler-state
through every call): the async pick-now/observe-later workflow with
out-of-order `update(uid, ...)` resolution has no benefit from
immutability — a dict keyed by opaque IDs is the right structure for "a
job handed out, not yet resolved," and none of that bookkeeping runs
inside `jit` anyway. One accepted tradeoff: `Sampler` objects aren't
directly `vmap`-able across whole samplers — acceptable, since nothing
today needs to batch across samplers, only across candidate points within
one sampler's `pick()`.

---

## 4. Coding practices & typing

### Sampler interface: `typing.Protocol`, not ABC

The coverage config (§6) already excludes `class .*\bProtocol\):` from
coverage requirements — the house style expects Protocol. It also kills
the current anti-pattern of `assert False` as a "must override" marker: a
missing implementation becomes a mypy structural-typing error, not a
runtime crash on first use.

```python
from typing import Protocol, runtime_checkable
from jaxtyping import Array, Float, PRNGKeyArray

@runtime_checkable
class Sampler(Protocol):
    """Active-sampling strategy over a bounded parameter space."""
    key: PRNGKeyArray

    def pick(self) -> tuple[Float[Array, "d"], str]:
        """Choose the next query location; return it with a job id."""
        ...

    def update(self, uid: str, y_true: Float[Array, "t"]) -> int:
        """Record an observed value for a pending job; return its index."""
        ...
```

### Shape typing: `jaxtyping`, no `beartype` inside `jit`

Annotate all public array signatures with `jaxtyping` (`Float[Array, "n d"]`)
for documentation/mypy value. Do not wrap `jit`-compiled hot paths in
`beartype` runtime shape checks — shapes must be static under tracing, and
per-call runtime validation on every retrace fights the tracer rather than
helping. Reserve `beartype` enforcement (if used at all) for the outer,
non-jitted API surface.

### State containers: plain pytree dataclasses

`types.py`'s `Range`/`Folds` become frozen, `jax.tree_util.register_dataclass`
dataclasses — no Equinox/Flax dependency by default. Escalate to Equinox
only if the architecture introduces NN-style learnable modules later.

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Range:
    """Hyperparameter bound triple: (lower, upper, initial value)."""
    lower: Float[Array, "d"]
    upper: Float[Array, "d"]
    initial: Float[Array, "d"]
```

Kernels become frozen pytree dataclasses implementing
`__call__(x_p, x_q) -> Array`, hyperparameters as array leaves (visible to
`grad`/`optax`/BlackJAX), structural config as static fields. Composition
(`a * k + k`) becomes operator overloading over a small kernel-node tree,
replacing the current string-keyed `compose`/`named_target` machinery.

### Acquisition functions: typed, not string-dispatched

Replace `acq_name: str` + a `dict`-lookup factory (`acq_defs()`) with a
first-class `AcquisitionFunction` Protocol; `Literal[...]` only at a
config-parsing boundary, converted immediately to the callable internally.

### Randomness: explicit `PRNGKey`, no global state

Replace `np.random.seed(seed)` globals with an explicit key threaded
through sampler state, split on every stochastic call
(`key, subkey = jax.random.split(self.key)`). `random_sample`/`grid_sample`
become `random_sample(key, lower, upper, n) -> Array`. This is a hard
requirement, not style: global RNG state is incompatible with
`jit`/`vmap` and with BlackJAX's own reproducibility model.

### Docstrings

NumPy-style headers stay (cheapest migration path, and `mkdocstrings`
supports `docstring_style: numpy` directly — no rewrite to Google style
needed). Cut verbosity: no `.. note:: [Properties Modified]` side-effect
logs (frozen dataclasses make these moot), no restating types mypy-strict
already enforces. Full docstrings stay mandatory on the public surface
(Protocol methods, kernels, `train`/`predict` entry points); private
helpers get one line or none, per the project's general terse-comment bias.

### Expected lint friction

- `ARG001` (unused args) will fire routinely at `lax.scan`/`lax.cond`
  callback sites (carried-but-unused state) — expected, acceptable noise,
  handle with `_`-prefixed names or targeted `# noqa`.
- Branch-elimination refactors (`if x_q is None: ...` guards, present in
  every kernel) are safe as long as the branch resolves on a static Python
  `None`, not a traced value — becomes a real porting risk (needs
  `lax.cond`) only if a later refactor makes that guard depend on a traced
  array.
- No significant false-positive risk expected from `SIM`/`TCH`/`UP`/`B008`
  against JAX idioms specifically.

---

## 5. Tooling & project setup

Target layout follows `jesserobertson/vibe-py-cookiecutter`: flat package
directory (no `src/` layout), `pixi.toml` + `pyproject.toml`,
`scripts/{quality,dev,build,docs,test}.py` as Typer+Rich CLIs wrapping the
actual tools, `.pre-commit-config.yaml`, `.github/workflows/`, `CLAUDE.md`.

### `pixi.toml` (draft)

```toml
[project]
name = "dora"
version = "0.1.0"  # or dynamic via setuptools-scm
description = "Bayesian active sampling with JAX/BlackJAX"
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64"]  # win-64 dropped, see below

[dependencies]
python = ">=3.12,<3.13"
jax = "*"
jaxlib = "*"
blackjax = "*"
optax = "*"
numpy = "*"
scipy = "*"
typer = "*"
rich = "*"

[feature.dev.dependencies]
pytest = "*"
pytest-cov = ">=6.2.1,<7"
coverage = ">=7.0.0,<8"
hypothesis = "*"
mypy = "*"
ruff = "*"
pre-commit = ">=4.2.0,<5"
python-build = ">=1.2.2.post1,<2"
twine = ">=6.1.0,<7"
setuptools-scm = "*"

[feature.docs.dependencies]
mkdocs = "*"
mkdocs-material = "*"
mkdocstrings = "*"
mkdocstrings-python = "*"

[environments]
default = ["dev", "docs"]
dev = ["dev"]
docs = ["docs"]

[tasks]
quality = { cmd = "python scripts/quality.py", description = "Lint, format, typecheck, coverage" }
dev = { cmd = "python scripts/dev.py", description = "Environment setup/status/clean" }
build = { cmd = "python scripts/build.py", description = "Package build and publish" }
docs = { cmd = "python scripts/docs.py", description = "Docs build/serve/deploy" }
test = { cmd = "python scripts/test.py", description = "Unit/integration/parity tests" }
check-all = { cmd = "python scripts/test.py all && python scripts/quality.py check", description = "Everything CI runs" }
```

Every package above resolves on conda-forge; no `[pypi-dependencies]`
fallback is expected for this stack. If GPJax is ever adopted, it's also
on conda-forge.

**GPU:** default environment stays CPU-only. Document a GPU opt-in
(separate `feature.gpu` environment via pip wheels — Google's own CUDA
wheels move faster than conda-forge's rebuilds) rather than making GPU the
default multi-platform solve.

**Windows:** drop `win-64`. `jaxlib` has no conda-forge Windows build, and
upstream JAX doesn't officially support native Windows (recommends WSL2).
Document WSL2 as the supported path for Windows users instead of fighting
a platform JAX itself doesn't target.

### CI

`.github/workflows/ci.yml` via `prefix-dev/setup-pixi` (built-in
lockfile-keyed caching): matrix `[ubuntu-latest, macos-14]` running
`pixi run check-all`; separate `docs` job deploying to GitHub Pages on
`main`/tags. Drop the NLopt source-build step and the stale Travis badge
entirely.

### Pre-commit

`ruff-format` + `ruff --fix` from `ruff-pre-commit`, plus a **local** mypy
hook that shells to `pixi run -e dev mypy dora` (not the sandboxed
`mirrors-mypy` hook, which won't see the real project dependencies/type
stubs).

### Docs

Migrate to mkdocs-material + mkdocstrings-python. Set
`docstring_style: numpy` so existing docstrings don't need a style
rewrite. Mechanical work: convert embedded Sphinx RST directives
(`.. note::`) to Markdown admonitions (`!!! note`), convert the 5
narrative `.rst` pages to Markdown (`pandoc -f rst -t markdown` + cleanup),
drop `matplotlib.sphinxext.plot_directive` in favor of pre-generated
static images. Roughly a half-day to one-day job, not a rewrite.

### Versioning

Adopt `setuptools-scm` (derive version from git tags) instead of the
hardcoded `version='0.1'`. Gate `twine upload` behind tag-push in CI;
prefer PyPI Trusted Publishing (OIDC) over a stored API token if/when this
becomes a public release workflow.

### `CLAUDE.md` additions specific to this repo

- No Python control flow on traced values inside `jit` — use `lax.cond`/
  `lax.while_loop`/`lax.select`.
- No `.item()`/host callbacks inside `pick()`/`update()` hot paths — use
  `jax.debug.print` for debugging instead.
- PRNGKey convention: never reuse a key; samplers hold/split an explicit
  key, never rely on global RNG state.
- Static vs traced args: shapes/`dims` etc. must be `static_argnames` on
  jitted functions or passed as plain Python values outside the traced call.
- Pointer to `pixi run quality check`/`fix` and the mypy-strict settings so
  future sessions don't reintroduce untyped code.
- Note on the temporary `pixi run test parity` command (§6) and when it's
  safe to delete it.

---

## 6. Testing strategy

### Replace the golden-file regression test with a layered pyramid

Today's entire suite is one end-to-end test diffing `X`/`y`/`virtual_flag`
against golden `.npz` files — brittle to any legitimate algorithmic change
(nlopt→optax will change numbers) and gives zero localization when
something breaks. Replace with:

```
tests/
├── conftest.py                     # x64 enable, shared fixtures
├── unit/
│   ├── test_kernel.py              # symmetry, PSD, diagonal=1, per kernel
│   ├── test_linalg.py              # jitchol validity, jitter escalation
│   ├── test_predict.py             # noiseless-limit recovery, var >= 0
│   ├── test_train.py               # NLML/cross-val on hand-built cases
│   ├── test_base_sampler.py        # pick/update bookkeeping, out-of-order
│   ├── test_utils.py                # ArrayBuffer growth
│   └── test_sampling_utils.py      # random_sample/grid_sample bounds
├── integration/
│   ├── test_gp_sampler_smoke.py    # invariants, not exact arrays
│   └── test_delaunay_sampler_smoke.py
└── migration/
    └── test_parity.py              # temporary, deleted post-migration
```

Markers map directly to `scripts/test.py unit`/`integration`/`all`:
`unit`, `integration`, `slow`, `hypothesis` (already in the template
config) plus a new `parity` marker for the migration-only suite.

### Numerical parity harness (temporary)

`tests/migration/test_parity.py`: run the same fixed-seed small GP
problems through both the old `revrand`/`nlopt` path and the new
JAX/optax path, assert agreement in learned hyperparameters and
posterior mean/variance within `rtol=1e-4, atol=1e-6` (loosen to
NLL-at-convergence agreement if the optimizer paths land on different
equally-valid local optima). **Practical blocker:** this needs `nlopt`
and `revrand` installed alongside the new JAX stack simultaneously —
isolate in a dedicated `pixi run -e migration-parity test` environment,
never in the default dev environment. Delete this entire directory and
feature environment in the same PR that removes the legacy code path —
track this explicitly as a migration-completion checklist item.

### Hypothesis properties (3 concrete ones)

1. Gram matrices from stationary kernels are symmetric and PSD (via
   `jitchol` succeeding within a small jitter budget) for random point
   sets and positive lengthscales.
2. `random_sample`/`grid_sample` outputs always lie within `[lower, upper]`.
3. Posterior variance is always `>= -1e-8` (floating-point tolerance).

Use `@settings(deadline=None)` — first-call JIT compilation will blow
default per-example deadlines.

### JAX-specific pitfalls to guard against

- **jit masking bugs:** run core kernel/predict unit tests both eager and
  jitted (parametrized fixture), assert identical results.
- **float32 default:** JAX disables x64 by default; this library does
  Cholesky/NLML math where precision loss is a real risk and would also
  break parity vs the old float64 numpy/scipy path. Set
  `JAX_ENABLE_X64=1` at the process/env level (primary mechanism, since
  some JAX behavior locks in before fixtures run), with an autouse
  `conftest.py` fixture asserting it's actually enabled.
- **PRNGKey determinism:** fixed, hardcoded keys per test
  (`jax.random.PRNGKey(0)`), not a shared global seed.
- **Out-of-order `update()`:** zero current coverage of the documented
  "observations can return out of order" use case — add explicit tests:
  pick 3 points, resolve in scrambled order, assert correct index mapping.

### Coverage targets

- `regressors/gp/` and `active_sampling/` (excluding `pltutils.py`):
  **90%+**, enforced in CI via `pixi run quality coverage --fail-under`.
- `server/`: lower bar (~60-70%) — thin glue code, cheaper to eyeball-verify.
- `pltutils.py`: exempt, added to `[tool.coverage.run] omit`.

### Test data

Prefer inline, analytically-checkable fixtures (e.g. 1D GP, 2 training
points, closed-form posterior) over golden-file snapshots — these survive
optimizer changes since they hardcode kernel/hyperparameters rather than
depending on a learned fit. Retire `tests/data/ref_data_*.npz` once the
new smoke tests are landed and passing on invariants (shape/NaN/bounds),
not exact-array diffs.

---

## 7. Phased rollout

| Phase | Scope | Acceptance gate |
|---|---|---|
| **0 — Parity baseline** | Snapshot fine-grained golden values from the *current* code (per-kernel outputs, Cholesky outputs, predict outputs, learned hyperparameters) on fixed seeds — finer grained than today's single end-to-end diff | Golden values captured and committed |
| **1 — `regressors/gp` → JAX, wired in to replace `revrand`** | Port kernel/linalg/predict mechanically; replace `nlopt` with `optax`; add the missing stacked/multi-task NLML | Per-function parity against Phase 0 within tolerance (compare converged NLML/predictive accuracy, not exact hyperparameters, since BOBYQA vs autodiff-gradient optimizers won't land on identical points). Independently mergeable without touching `active_sampling` |
| **2 — `active_sampling` on the new core** | Rewire `gp_sampler.py` off `revrand` onto the new JAX core; `vmap`/`jit` acquisition scoring; leave `base_sampler.py`/`delaunay_sampler.py` untouched | Existing `tests/test_sampler.py` end-to-end golden-file diff passes (loosen only where nlopt→optax causes expected divergence) — the strongest available regression signal |
| **3 — BlackJAX fully-Bayesian mode (opt-in)** | NUTS-over-hyperparameters as an additive `inference='bayes'` option, not default — changes `predict()`'s return shape and per-`pick()` cost | Posterior recovery on synthetic data with known hyperparameters; latency benchmark vs MLE default (a synchronous active-sampling loop may not tolerate full-Bayesian cost per pick) |
| **4 — Server layer** | FastAPI + pydantic replacing Flask; typed request/response schemas mirroring the new kernel/config dataclasses | Integration tests against the same create/observe/predict/query contract as today |

Note on acquisition optimization: today's `pick()` does pure random-search
over ~500 i.i.d. candidates with numpy — there is no gradient-based
optimization currently. Phase 2's `vmap`/`jit` port is a pure speed win
with no behavior change (good parity-test target). A gradient-refinement
pass (`jax.grad` + `optax` L-BFGS polishing the best random candidate) is
a valid later enhancement but must be an explicit, separately-tested
opt-in since it changes numeric output — not a silent default swap.
(`jaxopt` is being folded into `optax` and shouldn't be taken on as a new
dependency.)

---

## 8. Open decisions for you

1. **Exotic kernels** (`non_stationary`/`tree`/`nonstat_rr`) — port in
   Phase 1, or defer indefinitely since nothing currently calls them?
2. **Delaunay sampler** — keep indefinitely, or deprioritize/drop if usage
   data shows it's rarely selected?
3. **Fully-Bayesian default** — confirm MLE (Phase 1/2) stays the default
   `predict()` behavior, with BlackJAX NUTS strictly opt-in (Phase 3),
   given the latency concern for a synchronous sampling loop.
4. **GPU support** — is GPU actually needed, or is this CPU-only in
   practice? Affects whether the GPU-opt-in pixi environment is worth
   building now vs deferring.
