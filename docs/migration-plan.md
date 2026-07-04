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
| Delaunay sampler | **Replaced with a pure-JAX inverse-distance-weighting (IDW) interpolant** | No JAX-native Delaunay triangulation exists — see below |
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

**Delaunay sampler — replaced, not ported.** Researched whether a pure-JAX,
GPU-accelerated Delaunay triangulation exists: it doesn't. GPU-accelerated
Delaunay triangulation is still an active algorithms-research problem
(e.g. gDel3D, Local DeWall — hybrid GPU/CPU repair algorithms, not
libraries), and the only tensor-framework-native implementation is
`torch_delaunay` (PyTorch, not JAX). Triangulation is inherently
sequential/combinatorial — it isn't the kind of computation `jit`/`vmap`/
autodiff apply to, so forcing scipy's `Delaunay`/Qhull into JAX isn't a
real option.

Recommendation: swap the *interpolation method*, not the triangulation
algorithm. Replace Delaunay + bilinear interpolation with
**inverse-distance-weighting (IDW)** implemented directly in
`jax.numpy`: weight training targets by an inverse power of distance from
the query point, normalize. No triangulation step at all, trivially
`vmap`-able over many query points, fully differentiable, and GPU-native
for free as an ordinary JAX computation — it plays exactly the role
Delaunay plays today (a cheap, non-Bayesian interpolant with no
hyperparameter training), without the topology-construction cost or an
external geometry dependency.

If IDW's purely-local interpolation proves too crude in practice, there's
a second pure-JAX fallback that reuses the GP core already being built:
kernel/RBF interpolation via a direct linear solve against a fixed kernel
(reusing the ported `linalg.py`/`predict.py`, skipping MLE training) —
same JAX/GPU-native properties, more accuracy at the cost of an O(n³)
solve as the training set grows. Start with IDW; only escalate to the
RBF fallback if IDW's accuracy is measurably insufficient.

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
| `active_sampling/delaunay_sampler.py` | **Replace with pure-JAX IDW interpolant** | No JAX-native Delaunay triangulation exists (only `torch_delaunay` for PyTorch); IDW fills the same "cheap non-Bayesian interpolant" role, GPU-native, no triangulation dependency |
| `server/server.py`, `response.py` | **Replace with FastAPI + pydantic** | Phase 4, not JAX-related |
| `revrand.legacygp` (external dep) | **Drop entirely** | Superseded by the promoted, ported `regressors/gp` |
| `nlopt` | **Drop** | Fully replaced by `optax` |
| `sklearn` (clustering only) | **Keep** | `make_folds` stays scipy-side |

---

## 3. Orchestration: fully immutable, functional state-threading

Revised from a hybrid "mutable shell around a jitted core" design after
discussion — the project prefers to keep immutability and a functional
style all the way out to the `Sampler` boundary, not just in the inner
math. `Sampler`/`GaussianProcess` state becomes a single frozen, pytree-
registered dataclass, and `pick`/`update` become pure functions that
return a new state rather than mutating one in place:

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplerState:
    X: Float[Array, "capacity d"]
    y: Float[Array, "capacity t"]
    virtual_flag: Bool[Array, "capacity"]
    n_filled: int                      # cursor into the preallocated buffers
    pending: dict[str, int]            # copied, never mutated, per call
    key: PRNGKeyArray

class Sampler(Protocol):
    def pick(self, state: SamplerState) -> tuple[SamplerState, Float[Array, "d"], str]: ...
    def update(self, state: SamplerState, uid: str, y_true: Float[Array, "t"]) -> SamplerState: ...
```

Callers (demo scripts, the REST server, notebooks) hold the current state
explicitly and thread it forward: `state, x, uid = sampler.pick(state)`.
The numerically heavy work (train/condition/predict/acquisition scoring)
is still `@jax.jit`-decorated pure functions operating on `state`'s array
fields, exactly as before — the change is that the *outer* orchestration
is now equally pure, not a mutable object wrapping a pure core.

Two things make this practical rather than merely principled:

- **Preallocated buffers, not append-and-copy.** `X`/`y`/`virtual_flag`
  get a fixed `capacity` up front (or grow by doubling — allocate a new,
  bigger array and copy once, the same amortized cost as the old
  `ArrayBuffer`'s doubling strategy, just expressed as "build new state"
  instead of "mutate in place"). Writes are `X.at[n_filled].set(row)` —
  the standard JAX functional-update idiom, not an O(n) append per call.
- **`pending` stays a plain dict, copied on write**, not reinvented as a
  persistent/immutable map. Outstanding job counts are small in practice
  (a handful to low hundreds at once), so `{**pending, uid: idx}` per
  `pick`/`update` is cheap. Not worth a specialized data structure for
  this.

**What this buys over the hybrid design:** every `SamplerState` is a
snapshot — checkpointing, replay, and time-travel debugging fall out for
free (want to see what the sampler would have picked 10 steps ago? you
already have that state object). It also makes the whole sampler
formally `vmap`-able across independent runs later, which a mutable
outer shell would foreclose.

**Cost, stated plainly:** callers must explicitly carry the returned
state forward instead of calling methods on a persistent object — a
small ergonomic tax, and arguably good practice for reproducibility
regardless. The async pick-now/observe-later workflow with out-of-order
`update(uid, ...)` resolution still works fine under this model: `pending`
is just a value living inside the immutable state, resolved by whichever
`update()` call names the right `uid`, in whatever order they arrive.

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
blackjax = "*"
optax = "*"
numpy = "*"
scipy = "*"
typer = "*"
rich = "*"
# jax/jaxlib deliberately NOT here — see [feature.cpu] / [feature.gpu] below

[feature.cpu.dependencies]
jax = "*"
jaxlib = "*"          # conda-forge CPU build

[feature.gpu.pypi-dependencies]
jax = { version = "*", extras = ["cuda12"] }  # pulls its own jaxlib; do not combine with feature.cpu

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
default = ["dev", "docs", "cpu"]
dev = ["dev", "cpu"]
docs = ["docs"]
gpu = ["dev", "gpu"]

[tasks]
quality = { cmd = "python scripts/quality.py", description = "Lint, format, typecheck, coverage" }
dev = { cmd = "python scripts/dev.py", description = "Environment setup/status/clean" }
build = { cmd = "python scripts/build.py", description = "Package build and publish" }
docs = { cmd = "python scripts/docs.py", description = "Docs build/serve/deploy" }
test = { cmd = "python scripts/test.py", description = "Unit/integration/parity tests" }
check-all = { cmd = "python scripts/test.py all && python scripts/quality.py check", description = "Everything CI runs" }
```

Every package except GPU `jax` resolves on conda-forge. If GPJax is ever
adopted, it's also on conda-forge.

**GPU: built in now, not deferred** (per your call — GPU support is
needed from the start, not a later add-on). JAX's own CUDA wheels
(`pip install "jax[cuda12]"`) move faster than conda-forge's CUDA
rebuilds, so GPU support goes through pixi's `[pypi-dependencies]` in a
dedicated `gpu` feature/environment (`pixi run -e gpu ...`), while a
separate `cpu` feature keeps the conda-forge `jaxlib` for anyone without
a GPU. **`cpu` and `gpu` are mutually exclusive alternatives**, not
additive layers — don't resolve both `jaxlib` (conda-forge) and
`jax[cuda12]` (pip) into the same environment, they'll conflict. That's
why `jax`/`jaxlib` were pulled out of the shared `[dependencies]` block
above and into `feature.cpu`/`feature.gpu` specifically.

**Open follow-up this creates:** GitHub-hosted Actions runners are
CPU-only, so `pixi run -e gpu ...` has no coverage in the CI matrix
sketched below unless a self-hosted GPU runner is wired up. Until that
exists, GPU correctness has to be verified locally before merging
anything that touches GPU-sensitive code paths — flag this as a real
operational gap, not just a config detail, since "GPU support" isn't
actually verified in CI otherwise.

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

**Delaunay → IDW migration** can run in parallel with any phase above —
it's a separate, self-contained sampler class, independent of the GP core
(Phase 1) and its sampler wiring (Phase 2). Recommend doing it alongside
Phase 1: it's small, has no triangulation-library dependency risk to
manage, and gives an early, low-risk JAX/GPU-native win while the harder
GP port is underway.

---

## 8. Decisions

Resolved in discussion:

1. **Exotic kernels** (`non_stationary`/`tree`/`nonstat_rr`) — **deferred
   indefinitely.** Not ported until a real caller needs them; keeps
   Phase 1 scope tight instead of solving a hard jit-compatibility problem
   for currently-dead code.
2. **Delaunay sampler** — **replaced with a pure-JAX IDW interpolant**
   (§1, §2), not kept scipy-based, since GPU-native was a hard requirement
   and no JAX Delaunay implementation exists to port to instead.
3. **Fully-Bayesian default** — **MLE/optax stays the default**
   `predict()` behavior; BlackJAX NUTS is strictly opt-in (Phase 3), given
   the latency cost of full posterior inference inside a synchronous
   pick-loop.
4. **GPU support** — **needed now, not deferred.** Built into the pixi
   setup as a first-class `gpu` feature/environment from the start (§5),
   with the CI-coverage gap (no GPU runner in the sketched Actions matrix)
   flagged as an explicit open follow-up rather than silently ignored.
5. **Sampler orchestration** — **fully immutable/functional state
   threading** (§3), not a mutable object wrapping a jitted core. Revised
   from the original hybrid recommendation after discussion.
