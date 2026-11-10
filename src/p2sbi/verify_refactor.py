"""
verify_refactor.py
------------------
Asserts that the config-driven simulator reproduces the pre-refactor hardcoded
simulator BIT-FOR-BIT under the baseline config, as CLAUDE.md requires.

Exact equality (np.array_equal), not np.allclose. A prior bound, phase centre
or spectral index transcribed wrong into the YAML produces a difference here;
in a trained posterior it would not be visible until far too late.

  conda run -n penv python -m p2sbi.verify_refactor --config configs/ilofar_J2145.yaml                # 200 seeds
  conda run -n penv python -m p2sbi.verify_refactor --config configs/ilofar_J2145.yaml --n_seeds 2000

Also checks (independently of the reference implementation):
  - cfg.prior_low / prior_high match the BoxUniform bounds sim_maker.py used
    to hardcode
  - cfg.reference_phase_bin == 297, the sentinel real_archive_loader.py needs
  - the assembled feature vector width equals cfg.x_dim
"""

import argparse

import numpy as np

from . import _reference_sim
from .pulsar_config import get_config
from .pulsarDMvary import sim_generator


# Values the pre-refactor pipeline hardcoded, for the config-only checks below.
# Transcribed from sim_maker.py:63-64 and real_archive_loader.py:26 as they
# stood before the refactor.
LEGACY_PRIOR_LOW  = np.array([-0.025, -4.0, -6.0, -0.08, np.log(7.0),   0.1], dtype=np.float32)
LEGACY_PRIOR_HIGH = np.array([ 0.025,  1.0,  1.0,  0.08, np.log(120.0), 0.5], dtype=np.float32)
LEGACY_PARAM_NAMES = ["DM", "gamma_leading", "gamma_main", "toa", "log_snr", "amp_ref"]
LEGACY_SENTINEL_BIN = 297
LEGACY_X_DIM = 20560


def assemble_x(sim):
    """Exactly the concatenation sim_maker.py performs."""
    return np.concatenate([
        sim["portrait"].flatten(),
        sim["flux_per_channel"],
        sim["centroids"],
    ])


def check_config_only(cfg):
    """Config-level regression guards. Independent of the reference sim."""
    failures = []

    if not np.array_equal(cfg.prior_low, LEGACY_PRIOR_LOW):
        failures.append(f"prior_low  {cfg.prior_low} != legacy {LEGACY_PRIOR_LOW}")
    if not np.array_equal(cfg.prior_high, LEGACY_PRIOR_HIGH):
        failures.append(f"prior_high {cfg.prior_high} != legacy {LEGACY_PRIOR_HIGH}")
    if cfg.param_names != LEGACY_PARAM_NAMES:
        failures.append(f"param_names {cfg.param_names} != legacy {LEGACY_PARAM_NAMES}")
    if cfg.reference_phase_bin != LEGACY_SENTINEL_BIN:
        failures.append(
            f"reference_phase_bin {cfg.reference_phase_bin} != legacy "
            f"MAIN_PHASE_BIN {LEGACY_SENTINEL_BIN} — TRAP 3: the simulator and "
            f"real_archive_loader.py would disagree on the dead-channel sentinel"
        )
    if cfg.x_dim != LEGACY_X_DIM:
        failures.append(f"x_dim {cfg.x_dim} != legacy {LEGACY_X_DIM}")

    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None,
                    help="Config to verify. Must be configs/ilofar_J2145.yaml: the frozen "
                         "_reference_sim.py reproduces only that setup")
    ap.add_argument("--n_seeds", type=int, default=200,
                    help="Number of seeds to compare (default 200)")
    args = ap.parse_args()

    cfg = get_config(args.config)

    print("=" * 66)
    print("  Refactor verification — exact equality against _reference_sim.py")
    print("=" * 66)
    print(f"  Config      : {cfg.path}")
    print(f"  Fingerprint : {cfg.fingerprint}")
    print(f"  Seeds       : {args.n_seeds}")
    print("-" * 66)

    # ---------------- config-only guards ----------------
    cfg_failures = check_config_only(cfg)
    if cfg_failures:
        print("  Config checks: FAILED")
        for f in cfg_failures:
            print(f"    - {f}")
    else:
        print("  Config checks: pass  "
              f"(priors, param order, sentinel bin {cfg.reference_phase_bin}, "
              f"x_dim {cfg.x_dim})")

    # ---------------- bit-for-bit comparison ----------------
    # Each seed is applied identically to both paths. Both consume np.random in
    # the same order, so identical seeds must give identical output.
    mismatches = []
    first_diff = None

    for i, seed in enumerate(range(args.n_seeds)):
        np.random.seed(seed)
        ref = next(_reference_sim.sim_generator(1))

        np.random.seed(seed)
        new = next(sim_generator(1, config_path=cfg.path))

        # theta: reference yields named scalars; config yields the ordered vector
        theta_ref = np.array([
            ref["DM"], ref["gamma_leading"], ref["gamma_main"],
            ref["toa"], np.log(ref["snr"]), ref["amp_ref"],
        ])
        theta_new = np.asarray(new["theta"])

        checks = {
            "portrait":         (ref["portrait"],         new["portrait"]),
            "flux_per_channel": (ref["flux_per_channel"], new["flux_per_channel"]),
            "centroids":        (ref["centroids"],        new["centroids"]),
            "mean_profile":     (ref["mean_profile"],     new["mean_profile"]),
            "theta":            (theta_ref,               theta_new),
            "x":                (assemble_x(ref),         assemble_x(new)),
        }

        bad = [k for k, (a, b) in checks.items() if not np.array_equal(a, b)]
        if bad:
            mismatches.append((seed, bad))
            if first_diff is None:
                first_diff = (seed, bad, checks)

        if (i + 1) % 50 == 0:
            print(f"    {i + 1} / {args.n_seeds} seeds compared "
                  f"({len(mismatches)} mismatched)")

    print("-" * 66)

    if not mismatches and not cfg_failures:
        print(f"  PASS — {args.n_seeds}/{args.n_seeds} seeds identical, bit for bit.")
        print(f"         portrait, flux, centroids, mean_profile, theta, x all")
        print(f"         exactly equal. Feature width {cfg.x_dim}.")
        print("=" * 66)
        return 0

    print(f"  FAIL — {len(mismatches)}/{args.n_seeds} seeds differ.")
    if first_diff:
        seed, bad, checks = first_diff
        print(f"\n  First mismatch at seed {seed}, fields: {bad}")
        for k in bad:
            a, b = checks[k]
            a, b = np.atleast_1d(a), np.atleast_1d(b)
            if a.shape != b.shape:
                print(f"    {k}: SHAPE differs — ref {a.shape} vs new {b.shape}")
                continue
            d = np.abs(a.ravel() - b.ravel())
            j = int(np.argmax(d))
            print(f"    {k}: max|diff| = {d.max():.3e} at flat index {j}  "
                  f"(ref {a.ravel()[j]!r} vs new {b.ravel()[j]!r})")

        print("\n  Interpretation:")
        print("    theta differs        -> a prior bound or the inferred/nuisance")
        print("                            declaration ORDER differs from the")
        print("                            original draw sequence.")
        print("    portrait differs     -> a phase centre, spectral index,")
        print("                            concentration, or the band/reference")
        print("                            frequency was transcribed wrong.")
        print("    centroids only       -> the dead-channel sentinel bin differs")
        print("                            (TRAP 3).")
        print("    tiny max|diff|       -> still a real failure: the check is")
        print("                            exact by design. Do not relax it to")
        print("                            allclose; find the transcription bug.")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
