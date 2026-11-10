# SBI Pulsar Simulator — code reference

Simulation-based inference (SNPE-C / neural spline flows) for profile-domain
pulsar timing. Everything instrument- and pulsar-specific lives in a single YAML
config; no script contains a telescope or pulsar literal.

**Related documents**

| file | contents |
|---|---|
| `CLAUDE.md` | architecture, the known traps, resource budget for the cluster |
| `REFACTOR_PLAN.md` | the change plan with IDs `S0.1`…`S5.8`, and open items |
| `CHANGES.md` | every change made, with the reasoning and the bugs found |
| `README.md` | this file — what each script is and how to run it |

---

## 0. Installation
Just do `pip install p2sbi`
PSRCHIVE has to come from conda and is only needed for the real-archive stages
every stage needs --config path/to/file.yaml, because configs aren't included.
An example config is added with the install, once can see and use the pulsar_config script to check against the installation.


## 1. Quick start

```bash
# validate a config and see one specimen parameter draw
conda run -n penv python -m p2sbi.pulsar_config configs/german_J2145.yaml

# prove the simulator still reproduces the pre-refactor one bit-for-bit
conda run -n penv python -m p2sbi.verify_refactor --n_seeds 300

# ten sample simulations with diagnostic plots
for s in $(seq 1 10); do
  conda run -n penv python -m p2sbi.simulation_maker \
    --config configs/german_J2145.yaml --outdir sim_check --seed $s --verbose
done

# do the simulations look like real data?
conda run -n penv_psr python -m p2sbi.compare_sim_real archives/DE601/*.ar \
  --config configs/german_J2145.yaml --n 10 --outdir sim_check

# full pipeline
conda run -n penv python -m p2sbi.run_pipeline \
  --config configs/german_J2145.yaml --all --n_sims 100000
```

## 2. Environments

| env | contents | used by |
|---|---|---|
| `penv` | numpy, torch, sbi, astropy, matplotlib | everything except archive reading |
| `penv_psr` | the above **plus psrchive** | reading `.ar` archives |

`real_archive_loader.py` imports psrchive at the top and **requires** `penv_psr`.
`fit_vonmises.py`, `channel_grid_report.py` and `compare_sim_real.py` import it
lazily, so they only need `penv_psr` when the input is a `.ar` file — given a
`.pt` they run in `penv`.

## 3. The config system

One YAML per (pulsar, telescope). `inferred` **is** the theta vector in
declaration order — changing the order or the set invalidates existing
posteriors. `nuisance` parameters are sampled and marginalised but never
inferred. Components reference sampled parameters with `$name`.

```yaml
instrument:
  n_freq_channels:    61
  n_phase_bins:       512
  reference_freq_mhz: 153.80859375     # amplitude_ref/concentration_ref are AT this
  channel_freqs_mhz: [...]             # true centres; omit for a uniform linspace
  centroid_mode:      circular         # linear | circular

inferred:
  - {name: DM,  low: -0.025, high: 0.025, label: "$\\Delta$DM", unit: "pc cm$^{-3}$"}
  - {name: snr, low: 5.0, high: 120.0, log: true, display: exp}   # theta stores log(snr)

components:
  - name: Main
    amplitude_ref:      1.0
    phase_center_ref:   0.98065834
    concentration_ref:  $conc_main
    spectral_index_amp: $gamma_main
```

### Dead channels (RFI injection)

Real archives arrive with zapped channels that PSRCHIVE has set to exactly
zero, and `real_archive_loader.py` passes them on as zero portrait rows with
flux 0 and centroid 0. An optional `dead_channels:` block makes the simulator
produce the same thing, so the network learns that a zero row means *missing*
rather than reading it as a measurement of zero flux. Without the block, no
channels are zeroed and the simulator is bit-identical to before.

```yaml
dead_channels:
  site_channels:                 # measured: fraction of real epochs each channel is dead
    - {mhz: 178.42, p: 0.9995}   # MHz-addressed, snaps to the nearest channel
  bad_day_sigma: 1.0             # shared per-epoch shift: RFI comes in bursts
  generic:                       # support beyond the measured channels
    p_any:  0.10
    n_mean: 2.0
    modes:  {random: 0.5, block: 0.5}
    block_width_mhz: [2.0, 12.0]
  max_frac: 0.25                 # cap on ordinary dead channels
  wipeout:  {p: 0.005, frac: [0.3, 0.7]}
```

- **Measure `site_channels` from your own archives**: the fraction of epochs
  in which each channel's loader output has zero flux. Never use `p: 1.0`: it
  removes "channel alive" from training altogether.
- **For an uncharacterised telescope**, use `generic` alone with a large
  `p_any`.
- **Requires `centroid_mode: circular`.**
- **The block is part of the fingerprint.**
- **The mask distribution does not bias the posterior**, because the mask is
  visible in `x` and independent of θ. It only sets how much training data
  each pattern gets.

See `configs/german_J2145.yaml` for a measured example and
`configs/example_generic_pulsar.yaml` for a template. With the block present,
`ChannelSetEncoder` also excludes all-zero rows from attention and pooling.

### Inheritance

`extends:` deep-merges this file onto a base. **Mappings merge key by key;
lists are replaced, never concatenated** — `inferred` is the theta ordering and
`components` is the forward model, so a partially merged list would silently
produce a parameter vector nobody wrote down.

```yaml
extends: german_J2145.yaml
observatory:
  name:      DE601
  lat_deg:   50.523
  lon_deg:   6.8837
  height_m:  358.0
```

### The fingerprint

`cfg.fingerprint` is a 12-hex hash over the inference-relevant fields. It is
written into `metadata.pt` (dataset) and `posterior_meta.pt` (model) and checked
on resume and before training. It is what makes "wrong config for this dataset"
an error rather than silently wrong posteriors.

**Excluded on purpose:**
- `observatory.*` — Doppler correction is post-inference, so stations sharing a
  frequency setup share one fingerprint and one trained model
- `pulsar.dm_header` / `dm_conv` / `raj` / `decj` / `f0_par_hz` — post-inference
- parameter `label` / `unit` / `display` — cosmetic; editing a plot label must
  not invalidate a 20M-sim run

### Configs present

| file | setup | notes |
|---|---|---|
| `ilofar_J2145.yaml` | I-LOFAR IE613, 40 ch × 512 bins, 112–190 MHz, `centroid_mode: linear` | baseline; the trained 20M model belongs to this fingerprint |
| `german_J2145.yaml` | German stations, 61 ch × 512 bins, 118.08–189.61 MHz, ref 153.80859375, `centroid_mode: circular` | base for the six station files |
| `german_DE601.yaml` | Effelsberg, 50.5230 N, 6.8837 E, 358 m | `extends:` the base |
| `german_DE602.yaml` | Unterweilenbach, 48.501455 N, 11.287696 E | `extends:` the base |
| `german_DE603.yaml` | Tautenburg, 50.980111 N, 11.711167 E, 341 m | `extends:` the base |
| `german_DE604.yaml` | Bornim, 52.438 N, 13.016 E | `extends:` the base |
| `german_DE605.yaml` | Jülich, 50.897 N, 6.424 E | `extends:` the base |
| `german_DE609.yaml` | Norderstedt, 53.698454 N, 9.969239 E | `extends:` the base |
| `example_second_station.yaml` | 24 ch × 256 bins test fixture | component values are **placeholders** — re-fit before science use |
| `example_generic_pulsar.yaml` | 253 ms pulsar, L-band, 32 ch × 1024 bins, 3 components, **7 inferred parameters with free scattering** | test fixture proving the pipeline is genuinely config-driven — see §12 |

All six station files carry only an `observatory:` block, so they report the
**same fingerprint as the base** and share one dataset and one trained model.
Heights are 0 where not supplied; the effect on the Doppler correction is
4.9e-10 pc cm⁻³ for 358 m, i.e. irrelevant.

---

## 4. Pipeline order

```
configs/*.yaml
     │
     ├─ sim_maker.py ──────────► <dataset>/sim_chunks/*.pt + metadata.pt
     │                                      │
     │                           train_sbi.py ──► posterior.pt + posterior_meta.pt
     │                                      │
     ├─ simulation_maker.py ──► one test sim .pt (+ plots)
     │                                      │
     │                        DM_sampling.py ──► corner plot, parameter table
     │                        PPC.py         ──► posterior predictive checks
     │                        pp_tests.py    ──► PP calibration
     │                        simul_valid.py ──► true-vs-recovered grid
     │
     └─ real_archive_loader.py ─► real_*.pt ──► timeseries.py ──► timeseries.csv
                                                     │
                                       profile_evolution.py, toa_diagnostics.py
```

`run_pipeline.py` orchestrates the simulation branch.

---

## 5. Core library (imported, no CLI)

### `pulsar_config.py`
The config loader. Parses and validates YAML, resolves `$param` references and
`extends:` chains, computes the fingerprint.

`load_config()` is `lru_cache`d, so passing a config **path** (a picklable
string) into multiprocessing workers is safe — each worker parses the YAML once.

Key accessors: `n_chan`, `n_bins`, `freqs_mhz`, `ref_freq_mhz`, `x_dim`,
`period_ms`, `period_s`, `prior_low`, `prior_high`, `param_names`,
`param_labels`, `reference_phase`, `reference_phase_bin`, `centroid_mode`,
`epoch_mjd`, `dm_param`, `snr_param`, `geodetic`, `fingerprint`,
`display_specs()`, `sample()`, `theta_vector()`, `build_component_kwargs()`,
`scattering_values()`, `summary()`.

`_validate()` fails loudly on: duplicate parameter names, unresolvable `$refs`,
phase centres outside [0,1), a reference frequency outside the band, an
inconsistent explicit channel grid, a bad `centroid_mode`, and a `linear`
centroid mode where the pulse's reach crosses the phase wrap.

Has a CLI for convenience:
```bash
conda run -n penv python -m p2sbi.pulsar_config [configs/x.yaml]
```
Prints `summary()`, one specimen draw, the theta vector and the resolved
component kwargs. **Run this first whenever you edit a config.**

### `features.py`
**The single definition of the feature vector**, `x = portrait.flatten() |
flux_per_channel | centroids`. Called by both the simulator and
`real_archive_loader.py`.

This module exists because traps 3 and 4 in `CLAUDE.md` — the dead-channel
sentinel and the portrait normalisation — used to be implemented twice and kept
in step by hand. Divergence is now structurally impossible.

`normalise_portrait()`, `flux_per_channel()`, `channel_centroids()`,
`build_features()`. Depends on numpy only, so `penv_psr` can import it without
torch or sbi.

**`centroid_mode`:**
- `linear` — flux-weighted mean bin index with the out-of-range sentinel. The
  pre-2026 definition; keeps the I-LOFAR dataset and model valid.
- `circular` — first Fourier moment as a wrapped offset from
  `reference_phase_bin`. Required whenever the pulse sits near phase 0 or 1: the
  linear form is a linear statistic on a circular quantity, and on the German
  config it is non-monotonic in DM and inverts the ToA response.

### `pulsarDMvary.py`
The simulator and the encoders.

- `ProfileComponent` — one von Mises component with frequency/time evolution
- `PulsarProfileSimulator` — portrait generation; use `.from_config(cfg)`
- `build_components(cfg, sampled)` — instantiate this config's components
- `sim_generator(n_sims, config_path=None)` — the training-data generator. Takes
  a **path**, not a config object, so it is safe inside `mp.Pool`
- `simulate_from_theta(theta, config_path, seed)` — one portrait with the
  inferred parameters fixed; used by PPC
- `_finalise_portrait(portrait, cfg)` — bandpass, then `features.build_features`
- `MultiBranchEncoder` (139,488 params) — fixed 2D CNN over the portrait
- `ChannelSetEncoder` (309,344 params) — per-channel tokens with an explicit
  `log(ν/ν_ref)` tag, self-attention across channels
- `build_encoder(name, cfg)`, `encoder_param_count()`, `get_device()`

**RNG ordering in `sim_generator` is load-bearing.** The draw sequence is
`cfg.sample()` (inferred then nuisance, in YAML order) → the noise realisation →
the bandpass lognormal. Reordering the `inferred:` or `nuisance:` blocks changes
every simulation from a given seed.

### `_reference_sim.py` — **FROZEN, DO NOT EDIT**
Verbatim copy of `sim_generator` and `J2145_components` as they were before the
config refactor. Exists only so `verify_refactor.py` can assert bit-for-bit
equality. Editing it defeats its entire purpose.

### `efficiency.py`, `natastro_style.py`
Support libraries: training-efficiency bookkeeping (`ChunkStats`, `RoundStats`,
`compute_round_efficiency`, `get_hardware_profile`) and the matplotlib
publication style (`set_style`, `NAT_W_SINGLE`, `NAT_W_DOUBLE`).

---

## 6. Pipeline stages

### `run_pipeline.py` — orchestrator

```bash
conda run -n penv python -m p2sbi.run_pipeline --config configs/german_J2145.yaml \
    --all --n_sims 100000 --chunk_size 10000
```

**Stages:** `--make-dataset` `--train` `--simul-valid` `--make-test` `--sample`
`--PPC` `--pp-test` `--val-loss-plot` `--all`

**Options:** `--config` `--encoder` `--dataset` `--n_sims` `--chunk_size`
`--n_workers` `--epochs` `--batch_size` `--sims_per_round` `--seed`
`--posterior` `--test_sim` `--n_test_cases` `--n_valid` `--n_valid_samples`
`--resume` `--total_n_sims` `--test_seed` `--set NAME=VALUE` (repeatable)

### Resource passthroughs

All three default to `None`, meaning "defer to the downstream script". Omitting
them reproduces the previous command byte for byte, so none of them changes
existing behaviour.

| flag | forwarded as | default if omitted |
|---|---|---|
| `--n_workers` | `sim_maker.py --n_workers` | `os.cpu_count()` |
| `--batch_size` | `train_sbi.py --batch_size` | 4096 on CUDA, 256 on CPU/MPS |
| `--sims_per_round` | `train_sbi.py --group_size` = `sims_per_round // chunk_size` | 100,000 sims/round |

**`--n_workers` on SLURM: set it explicitly to `$SLURM_CPUS_PER_TASK`.**
`os.cpu_count()` reports the whole node, not your cgroup allocation, so the
default oversubscribes a shared node.

**`--batch_size`**: the CUDA default of 4096 can OOM on a memory-constrained
GPU. There is no way to override it through `run_pipeline.py` without this flag.

**`--sims_per_round`**: expresses the GPU-resident budget in simulations rather
than in chunk files, which is the quantity VRAM actually limits. It applies only
in chunk mode (`torch.cuda.is_available()`), and the conversion is floored at 1
so a `chunk_size` larger than the target still yields a valid group. With
`--chunk_size 10000`:

| `--sims_per_round` | `group_size` | sims/round |
|---|---|---|
| unset | 10 | 100,000 |
| 30000 | 3 | 30,000 |
| 5000 | 1 | 10,000 (one chunk is the floor) |

When left unset on CUDA the run prints a warning, because 100,000 is a
historical value measured on different hardware and is not a safe assumption for
a new GPU or a config with a wider feature vector. Verify empirically before
relying on it — note the help text points at `tools/probe_vram.py`, which is
**not** part of this repository.

Resolves and validates the config up front so a typo fails immediately rather
than six hours in. Threads `--config` into `sim_maker` and `simulation_maker`;
the others default to the config recorded in the dataset or posterior sidecar,
which cannot drift. `--PPC` requires `--sample`.

`--set` pins a test-simulation parameter and is validated against the config's
declared parameters.

### `sim_maker.py` — training dataset

```bash
conda run -n penv python -m p2sbi.sim_maker --config configs/german_J2145.yaml \
    --n_sims 30000000 --chunk_size 10000 --outdir $SCRATCH/sims --n_workers 32
```

`--config` `--n_sims` `--outdir` `--chunk_size` `--n_workers` `--force`

Writes `sim_chunks/sims_chunk_NNNN.pt` (`theta`, `x`) and `metadata.pt`
(prior, `config_path`, `config_yaml` verbatim, `config_fingerprint`,
`param_names`, `param_labels`, `log_params`, `n_chan`, `n_bins`, `x_dim`).

Builds the `BoxUniform` prior from `cfg.prior_low`/`prior_high` — the **same**
config object the simulator samples from, so a prior is declared exactly once.

Resumes automatically from existing chunks, and **aborts** if their fingerprint
differs from the config given (`--force` overrides). Without that guard,
resuming with an edited YAML would silently produce a dataset that is half one
instrument and half another.

### `train_sbi.py` — training

```bash
conda run -n penv python -m p2sbi.train_sbi --dataset $SCRATCH/sims \
    --mode single_pass --encoder channelset
```

`--dataset` `--config` `--encoder {channelset,multibranch}` `--mode
{auto,single_pass,chunk}` `--epochs` `--batch_size` `--stop_after`
`--group_size` `--lr` `--resume` `--seed` `--training_log` `--force`
`--wait_for_chunks` `--poll_interval` `--stall_timeout`

Needs **no** `--config`: it reads the one recorded in `metadata.pt`, so it cannot
drift from the data, and fingerprint-checks before training.

Derives the expected feature width from the config and **aborts** on a mismatch
rather than letting an unchecked width through to `append_simulations`. Legacy
chunks carrying the old `centroid_residuals` block are detected and trimmed.

Writes `posterior.pt` and `posterior_meta.pt` (fingerprint, `param_names`,
`param_labels`, `log_param_indices`, encoder name and parameter count, prior
bounds, seed). Downstream scripts read the sidecar instead of hardcoding.

> **`--mode single_pass` loads the whole dataset into CPU RAM.** For the German
> config that is 122.5 KiB per simulation, so 30M sims needs 3.76 TB. Above your
> node's RAM you must use chunk mode — see the resource table in `CLAUDE.md`.
>
> Memory-mapping the chunks does **not** escape this. Tested on sbi 0.25.0:
> `append_simulations` copies into its own tensor — `inference._x_roundwise`
> has a different `data_ptr` from a memmap-backed input — so the data becomes
> RAM-resident regardless. Escaping the RAM ceiling would mean replacing sbi's
> `train()` with a custom loop.

#### Streaming: overlapping generation and training

`--wait_for_chunks` makes chunk mode start on the chunks that already exist and
**wait** for the rest as `sim_maker.py` writes them, so the CPU generation job
and the GPU training job overlap instead of being chained with
`--dependency=afterok`.

```bash
# submit both at once; no --dependency
sbatch --cpus-per-task=32 --wrap="... sim_maker.py --config <cfg> \
    --n_sims 30000000 --chunk_size 10000 --outdir $SCRATCH/sims --n_workers 32"

sbatch --gres=gpu:1 --wrap="... train_sbi.py --dataset $SCRATCH/sims \
    --mode chunk --group_size 3 --batch_size 1024 \
    --wait_for_chunks --poll_interval 60 --stall_timeout 3600"
```

How it works: `sim_maker.py` records `expected_chunks` in `metadata.pt`
**before generating anything**, so training knows how many rounds to plan. Each
round blocks until its chunk files appear, then trains exactly as before.

Three things make it safe:

- **Atomic chunk writes.** `sim_maker.py` writes `<chunk>.pt.tmp` then
  `os.replace()`s it. `torch.save()` streams into its destination, so a poller
  opening a file mid-write would read a truncated pickle. `os.replace` is atomic
  within a filesystem, so a chunk is either absent or complete. *This matters
  even without streaming*: a generation job killed mid-write would otherwise
  leave a truncated chunk that a later training run loads as real data.
- **`--stall_timeout`** (default 1800 s) aborts with a clear error if no new
  chunk appears. The two jobs run on different Slurm clusters, so `squeue` here
  cannot see the generator — "no new chunk for N seconds" is the only available
  proxy for "the producer died". Progress *anywhere* in the directory resets
  the clock, so a generator that is alive but behind does not trip it.
- **Incompatible with `single_pass`**, which is rejected with an explanation:
  single_pass hands the whole dataset to sbi at once, so there is nothing to
  stream into.

**What it actually buys.** Generation is 2–5 h for 30M. Training in chunk mode
is far longer, so the saving is the head start, not a halving:

| sims/round | VRAM for the group | rounds at 30M | at 12 min/round | at 25 min/round |
|---|---|---|---|---|
| 30,000 | 3.5 GB | 1,000 | 200 h | 417 h |
| 100,000 | 11.7 GB | 300 | 60 h | 125 h |
| 200,000 | 23.4 GB | 150 | 30 h | 62 h |

The group is moved to the **GPU** (`append_simulations(theta.to(device),
x.to(device))`), so sims/round is bounded by VRAM, not host RAM. Raising it is
by far the bigger lever on wall clock — and on the catastrophic-forgetting
exposure, since it is 1,000 sequential warm-started rounds that create the risk
`CLAUDE.md` flags for chunk mode.

### `simulation_maker.py` — one test simulation

```bash
conda run -n penv python -m p2sbi.simulation_maker --config configs/german_J2145.yaml \
    --outdir sim_check --seed 7 --DM 0.01 --snr 40 --verbose
```

`--config` `--outdir` `--seed` `--mjd` `--verbose` `--nopt` `--noplot`
`--justplot` `--sim`, **plus one flag per config parameter**

The per-parameter flags are **generated from the config**: `--DM`,
`--gamma_main`, `--conc_main` … exist because the YAML declares them. Point it
at a different config and the flags follow. Unset parameters are drawn from
their priors.

Saves a `.pt` with the portrait, clean portrait, per-component portraits, flux,
centroids, mean profile, **every** declared parameter in `true_params`, the
ordered `true_theta`, and metadata including the config fingerprint. Plus a
portrait plot and a 4-panel diagnostic.

`--justplot` regenerates plots from an existing `.pt`.

### `real_archive_loader.py` — real data → `.pt` *(needs `penv_psr`)*

```bash
conda run -n penv_psr python -m p2sbi.real_archive_loader path/to.ar \
    --config configs/german_DE601.yaml --outdir real_data
```

positional `archive`; `--config` `--outdir` `--noplot` `--verbose`

Validates the archive against the config (phase bins, channel count divisibility,
both band edges within `freq_tol_mhz`) and **exits non-zero** on failure.
Then tscrunches, pscrunches, removes the baseline, dedisperses, fscrunches to
`n_freq_channels`, and flips to ascending channel order if needed.

Derives features through `features.build_features` — the same call the simulator
makes, which is what keeps traps 3 and 4 closed.

Records each observation's **true channel frequencies** and the induced ΔDM scale
error in the saved metadata, and warns when that exceeds 1e-4 pc cm⁻³. Without
this the training/inference frequency mismatch is invisible downstream.

---

## 7. Verification

### `verify_refactor.py`
```bash
conda run -n penv python -m p2sbi.verify_refactor --n_seeds 300 [--config ...]
```
Asserts the config-driven simulator reproduces `_reference_sim.py`
**bit-for-bit** — `np.array_equal`, not `allclose` — on portrait, flux,
centroids, mean profile, theta and the assembled `x`. Also checks the priors
against the legacy `BoxUniform` bounds, the parameter order, the sentinel bin
and `x_dim`.

**Keep it exact.** The `period_ms` transcription bug it caught was 5.3e-7
relative; `allclose` would have passed and shipped the wrong period. Run it once
on any new machine before a long job — it catches a numpy/scipy version changing
FFT or RNG behaviour for the cost of a minute.

Only valid with `centroid_mode: linear` (the default), since circular centroids
are deliberately a different feature.

### `fit_vonmises.py` — component fitting
```bash
# free fit (the measurement)
conda run -n penv_psr python -m p2sbi.fit_vonmises archives/DE601/*.ar \
    --config configs/german_J2145.yaml --outdir archives/DE601/ --max_components 6

# constrained to the config's decomposition, with a paste-ready YAML block
conda run -n penv_psr python -m p2sbi.fit_vonmises archives/DE601/*.ar \
    --config configs/german_J2145.yaml --seed_from_config \
    --emit-yaml --recentre-on Main
```
positional `infile` (PSRCHIVE `.ar` / `.fits` — same PSRFITS format — or `.pt`); `--config` `--outdir` `--max_components`
`--force_n_components` `--min_significance` `--ref_window` `--seed_from_config`
`--emit-yaml` `--recentre-on` `--scattering` `--spectral-indices` `--n-subbands`
`--nchan` `--ref-freq`

**`--scattering`** — for scattered pulsars. **Needs no config**: without
`--config` it reads phase bins, channel grid, period and DM from the archive
(`--nchan` scrunches it, `--ref-freq` sets where amplitudes, widths and tau are
quoted) and **writes the config itself** to `--outdir`: `<archive>_config.yaml`,
complete and loadable (it is loaded back to check), every value from the
archive header, tempo2's site list and the fit, and every prior from one stated
rule (`--prior-nsigma`, `--prior-floor`, `--prior-index-floor`, `--prior-dm`,
`--prior-tau-factor`, `--prior-alpha`, `--prior-toa`, `--prior-snr-factor`;
explained in the file's header). `<archive>_fit.yaml` beside it records every
fitted number with its error, the model-selection steps and the
tau-per-subband check. With `--config` it fits on the
config's grid instead and compares the result with the config's components and
priors. Either way it fits the whole portrait, not the band-averaged profile,
with the simulator's own model: von Mises
components with power-law amplitude and width in frequency, convolved with one
exponential tail `tau(nu) = tau_ref (nu/nu_ref)^alpha` (applied in the Fourier
domain, so it wraps), plus a residual DM referenced to the top channel. A gain
and baseline per channel are solved linearly at every step. The greedy + AIC
search is the same, except that each step tries the three best starting
positions (unused peaks, then the largest residual bumps) and keeps the best
fit. The significance scan additionally rejects a component whose
spectral index sits at its search limit, or that looks like a scattering tail
imitated by a von Mises (broader than a brighter component it trails and
brightening to low frequency much faster; with `--spectral-indices shared`,
over twice as broad). Without those checks, on J1939+2134 the search added 5
such terms and pulled tau from 48 to 33 us. Reports tau, alpha, dDM, tau per
subband against the power law, suggested `sigma_bandpass` and S/N, prior checks
including the scattering parameters, and with `--emit-yaml` a components block
plus the fitted scattering values. `--spectral-indices free` (default) gives each
component its own indices; `shared` one per pulse.

```bash
# no config: fit, then write <archive>_config.yaml and <archive>_fit.yaml to --outdir
conda run -n penv_psr python -m p2sbi.fit_vonmises archive.fits --scattering --nchan 64 --outdir out/

# against an existing config
conda run -n penv_psr python -m p2sbi.fit_vonmises archive.ar --config configs/<cfg>.yaml \
    --scattering [--seed_from_config --emit-yaml --recentre-on Main]
```

Fits von Mises components to the frequency-scrunched profile and reports:

- **AIC/whiteness model selection**, then a **significance scan** that rejects
  components which are below `--min_significance` × the off-pulse RMS, have κ at
  the fitter's ceiling, or sit inside a brighter component's half-width
  (`BLENDED`). AIC alone is easily gamed — on real data it preferred N=10 where
  only 4 components were detections.
- a **config-seeded cross-check** that always runs, comparing the free search
  against a fit seeded at the config's own geometry by AIC and RMS. If they reach
  the same optimum, the config's decomposition is what the data prefers
  *independently of having the config*.
- **component separations from the reference component** — the quantity that must
  be preserved across telescopes, since a constant offset is degenerate with `toa`
- **fitted values against the config's priors**, flagging anything outside or
  hugging a bound
- the phase offset against the `toa` prior, saying whether it is absorbed,
  marginal, or unreachable

Phases come from the frequency-scrunched fit (all channels, best S/N;
`spectral_index_center` is 0 so phase carries no frequency dependence); widths
and amplitudes from a `--ref_window` average around the reference frequency,
where the config defines them.

`--seed_from_config` only changes which fit gets the detailed write-up — the
free search runs and is reported regardless, so it cannot self-confirm.

---

## 8. Diagnostics for multi-station work

### `channel_grid_report.py`
```bash
conda run -n penv_psr python -m p2sbi.channel_grid_report \
    DE601=a1.ar DE601=a2.ar DE603=b1.ar ... \
    --config configs/german_J2145.yaml --nominal mean
```
positional `inputs` (`.ar` or `.pt`, optional `LABEL=` prefix); `--config`
`--nominal {mean,config}` `--dm_errorbar`

Decides whether stations or epochs can share one trained model, and emits the
nominal (mean) grid as a paste-ready `channel_freqs_mhz:` block.

Reports two effects separately, because they do different damage:
- the component **parallel** to 1/ν² is degenerate with ΔDM → a pure **scale
  error** on inferred DM
- the **orthogonal** remainder biases nothing but pushes the observation out of
  the training distribution; it is judged against the per-channel Cramér-Rao
  position bound across the S/N prior

With `LABEL=` groups and two or more epochs each, it also decomposes the scatter
into between- and within-station parts and says whether per-station channel
lists would help. A smooth offset is almost entirely DM-degenerate; random
per-channel scatter is the opposite and is irreducible by any fixed grid.

### `compare_sim_real.py`
```bash
conda run -n penv_psr python -m p2sbi.compare_sim_real archives/DE601/*.ar \
    --config configs/german_J2145.yaml --n 10 --outdir sim_check
```
positional `real` (`.ar` or `.pt`); `--config` `--n` `--outdir` `--seed` `--snr`

**Run this before committing to a large simulation run.** Overlays N simulated
profiles on a real archive: scrunched profile, on-pulse zoom with the config's
component phases marked, real vs simulated portrait, and flux versus frequency.

Matches the simulated S/N to the archive's measured off-pulse RMS and sets DM
and ToA to zero, so the comparison is about profile shape and spectral behaviour
rather than the parameters being inferred. The flux-versus-frequency panel is
how the `gamma_main` prior being too narrow was found.

---

## 9. Post-inference analysis

### `DM_sampling.py`
```bash
conda run -n penv python -m p2sbi.DM_sampling --posterior_tensor <dataset>/posterior.pt \
    --test_simulation <sim>.pt --outdir <dataset>
```
`--posterior_tensor` `--test_simulation` `--outdir` `--config` `--num_samples`

Samples the posterior (rejection sampling, falling back to masked direct
sampling if acceptance stalls) and produces a corner plot plus a median/MAP/σ
table. Labels, units, prior bounds and the display conversions all come from the
config; truths are read from `true_params` by name.

### `PPC.py`
```bash
conda run -n penv python -m p2sbi.PPC --posterior_tensor <p>.pt \
    --test_simulation <sim>.pt --outdir <d> --n_sims <N>
```
`--posterior_tensor` `--test_simulation` `--outdir` `--config` `--n_sims`
`--n_samples`

Posterior predictive checks: re-simulates each posterior sample through
`simulate_from_theta`, so a PPC cannot test against a different forward model
than the one trained.

### `pp_tests.py`
```bash
conda run -n penv python -m p2sbi.pp_tests --posterior_tensor <p>.pt \
    --outdir <d> --n_test_cases 100
```
`--posterior_tensor` `--outdir` `--config` `--n_sims` `--n_test_cases`
`--n_posterior_samples` `--sample_batch` `--just_plot`

PP (probability-probability) calibration over many independent test
simulations, with a KS p-value per parameter. **This is the check that catches
catastrophic forgetting** if you had to train in chunk mode.

### `simul_valid.py`
```bash
conda run -n penv python -m p2sbi.simul_valid --dataset <d> --posterior <p>.pt \
    --n_test 1000 --n_samples 5000
```
`--dataset` `--posterior` `--config` `--n_test` `--n_samples` `--sample_batch`
`--n_workers` `--outdir` `--seed` `--just_plot`

True-vs-recovered grid and normalised residuals, with bias and RMSE per
parameter. Writes `simul_valid_summary.csv`.

### `timeseries.py`
```bash
conda run -n penv python -m p2sbi.timeseries --outdir real_data \
    --posterior_tensor <p>.pt --config configs/german_DE601.yaml
```
`--outdir` `--posterior_tensor` `--config` `--n_samples` `--conventional_dm`
`--offset_method {inv_var,ml_pairs}` `--toa_leakage` `--just_plot`

Batch inference over `real_*.pt` files → `timeseries.csv` plus plots. This is
the **only** consumer of the observatory position, used for the annual Doppler
DM correction, and the reason station configs exist at all.

> Its output CSV schema is still specific to the J2145 six-parameter set (it
> writes columns like `gamma_leading_map`). The theta indices are looked up by
> name and fail loudly, but a different parameter set would need the schema
> generalising.

### `profile_evolution.py`, `toa_diagnostics.py`, `val_loss_plot.py`
```bash
conda run -n penv python -m p2sbi.profile_evolution --outdir real_data
conda run -n penv python -m p2sbi.toa_diagnostics --csv real_data/timeseries.csv
conda run -n penv python -m p2sbi.val_loss_plot --dataset <dataset>
```
`profile_evolution.py` (`--outdir` `--include_flagged` `--periodogram`) plots
γ_lead, γ_main and amp_ref against MJD from `timeseries.csv`.
`toa_diagnostics.py` (`--csv` `--f0` `--outdir` `--pepoch`) investigates excess
ToA scatter. `val_loss_plot.py` (`--dataset` `--training_log`) compares
validation-loss curves across runs; `train_sbi.py` invokes it automatically.

---

## 10. The traps

From `CLAUDE.md`, with where each is now enforced:

| # | trap | enforcement |
|---|---|---|
| 1 | `reference_freq_mhz` must lie in the band, and `amplitude_ref`/`concentration_ref` are defined **at** it | `_validate()` catches out-of-band; `fit_vonmises.py` reads the reference from the config so fitted values follow it |
| 2 | never declare priors twice | `sim_maker.py` builds `BoxUniform` from `cfg.prior_low/high`; there were **five** copies before |
| 2b | `inferred`/`nuisance` declaration order is load-bearing | `verify_refactor.py`; the fingerprint changes if you reorder |
| 3 | the centroid sentinel must match between training and inference | one implementation in `features.py`, called by both |
| 4 | portrait normalisation must match | same |
| 3/4 | dead channels must look the same in training and inference | the simulator zeroes dead rows **before** `build_features`, which is what PSRCHIVE delivers, so the loader's output and the simulator's are `array_equal` (`dead_channels:` block) |
| 5 | `period_ms` is part of what the model learned | `verify_refactor.py` exact equality |

## 11. Resource budget

Measured: `x_dim × 4` bytes and ~3 ms of one core per simulation.

| config | `x_dim` | per sim | 30M |
|---|---|---|---|
| I-LOFAR, 40 ch | 20560 | 80.3 KiB | 2.47 TB |
| German, 61 ch | 31354 | 122.5 KiB | 3.76 TB |

`--mode single_pass` needs that same figure in **CPU RAM**. Generation is
**I/O-bound**, not CPU-bound. Full tables in `CLAUDE.md`.

## 12. Any pulsar, any telescope — what a new config may change

**Yes: write a config and train.** Verified by running the whole pipeline on
`configs/example_generic_pulsar.yaml`, a fixture built to differ in every
dimension that could plausibly be hardcoded somewhere:

| | J2145 configs | generic fixture |
|---|---|---|
| period | 16 ms | **253 ms** |
| band | 110–190 MHz | **1200–1500 MHz** |
| shape | 61 ch × 512 bins | **32 ch × 1024 bins** |
| components | 4 | **3** |
| inferred parameters | 6 | **7** |
| scattering | fixed at 0 | **free, log-uniform, in theta** |
| noise model | `avg_peak` | **`radiometer`** |
| DM smearing | off | **on** |
| `spectral_index_center` | 0 everywhere | **non-zero on one component** |
| `sigma_broadening_bins` | 0 everywhere | **non-zero on one component** |

All seven pipeline stages pass. `theta` comes out `(n, 7)`, `x` `(n, 32832)`,
`metadata.pt` records `log_params: ['snr', 'tau_scatter']`, and `pp_tests.py`
produces a calibration curve for all seven parameters including
`log_tau_scatter`.

### Free scattering

```yaml
inferred:
  - {name: tau_scatter, low: 0.01, high: 8.0, log: true, unit: "ms", display: exp}

scattering:
  tau_ms:         {type: free, param: tau_scatter}
  spectral_index: {type: fixed, value: -4.0}
```

`type: free` names a declared parameter; `_validate()` rejects a name that does
not resolve. It can equally be a `nuisance` parameter if you want it
marginalised rather than inferred. `spectral_index` can be freed the same way.

### Constraints a new config must satisfy

These are enforced by `_validate()`, which lists **all** violations at once
rather than stopping at the first:

1. **A `toa` parameter must exist**, because `ComponentSpec.phase_offset_param`
   defaults to `"toa"`. To have a component *not* shift with ToA, set
   `phase_offset_param: null` on it. Omitting `toa` entirely gives
   `component 'X': phase_offset_param 'toa' is not declared`.
2. `reference_component` must name a declared component, and
   `reference_freq_mhz` must lie inside the band.
3. Every `$ref` must resolve; `simulation.dm_param` and `snr_param` must name
   declared parameters.
4. `phase_center_ref` in [0, 1); an explicit `channel_freqs_mhz` must be
   strictly ascending, the right length, and inside the band edges.
5. `centroid_mode: linear` is **rejected** if the pulse's reach (reference phase
   + DM sweep + ToA prior) crosses phase 0 or 1. Use `circular`.
6. `MultiBranchEncoder` needs `n_chan >= 8` (three `MaxPool2d(2)` layers halve
   the frequency axis). `ChannelSetEncoder` has no such limit.

### What is *not* yet generic

- **`timeseries.py`'s output CSV** writes J2145-specific columns
  (`gamma_leading_map` and so on). Its theta indices are looked up by name and
  fail loudly, so it will not silently mislabel — but a different parameter set
  needs the schema generalising. This is the only remaining parameter-set
  dependence in the pipeline.
- `verify_refactor.py` compares against the frozen I-LOFAR reference, so it is
  meaningful only for `ilofar_J2145.yaml`. For a new config the equivalent check
  is `compare_sim_real.py` against real data.
- `toa_diagnostics.py` and `profile_evolution.py` read `timeseries.csv` and
  inherit its schema.

> A bug this exercise found: `simul_valid.py` pre-allocated
> `np.zeros((n_test, 6))` — a literal 6 — so a 7-parameter config died with
> `could not broadcast input array from shape (7,) into shape (6,)`. Now
> `N_PARAMS`. It was the **only** hardcoded parameter count left, and it took a
> genuinely different config to surface it.

## 13. Adding a new telescope

1. Copy an existing config; set the frequency setup and observatory position.
2. Run `channel_grid_report.py` over one archive per station to get the nominal
   channel grid and check whether one model can serve them all.
3. **Re-fit the profile components at the new `reference_freq_mhz`** with
   `fit_vonmises.py` — trap 1. `_validate()` catches an out-of-band reference
   frequency; it cannot catch component values that are stale for one that moved.
4. Set `centroid_mode: circular` if the pulse sits near phase 0 or 1
   (`_validate()` will tell you).
5. `compare_sim_real.py` — confirm the simulations look like the data, and in
   particular that the real spectrum sits mid-envelope rather than at its edge.
6. `run_pipeline.py --config <new> --all --n_sims <N>`.

A different frequency setup means a different fingerprint, hence its own dataset
and model. If only the observatory changed, the fingerprint is unchanged: reuse
the model, and only `timeseries.py` differs.
