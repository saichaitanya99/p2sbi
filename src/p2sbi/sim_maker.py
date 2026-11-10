"""
sim_maker.py
------------
Generates the training dataset: chunked (theta, x) tensors plus a metadata.pt
describing exactly how they were made.

Everything instrument- and pulsar-specific comes from the YAML config. In
particular the prior is built from cfg.prior_low / cfg.prior_high — the SAME
config object the simulator draws from — so a prior is declared exactly once,
in the YAML. Declaring it twice is the bug this repo exists to remove.

  conda run -n penv python -m p2sbi.sim_maker --config configs/ilofar_J2145.yaml \
      --n_sims 100000 --outdir SBI_prods_100000 --n_workers 32
"""

import numpy as np
import torch
import argparse
import glob
import os
import math
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path
from sbi.utils import BoxUniform

from .pulsar_config import get_config, resolve_config_path, check_fingerprint
from .pulsarDMvary import sim_generator, get_device


def _sim_worker(seed, config_path=None):
    """Generate one simulation with the given RNG seed; called in each Pool worker.

    `config_path` is a string, not a PulsarConfig — a plain str pickles across
    the process boundary, and load_config() is lru_cached, so each worker parses
    the YAML exactly once no matter how many simulations it handles.
    """
    import torch
    torch.set_num_threads(1)   # prevent each worker spawning O(cpu_count) intra-op threads
    np.random.seed(seed)
    return next(sim_generator(1, config_path=config_path))


# _sim_worker must be at module level (above __main__ guard) so multiprocessing can pickle it.

def main(argv=None):

    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default=None,
                        help='Path to the YAML config. Required unless $SBI_PULSAR_CONFIG is set.')
    parser.add_argument('--n_sims',     type=int, default=10000)
    parser.add_argument('--outdir',     type=str, default=None,
                        help='Output directory. Default: sims_{pulsar}_{n_sims}')
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--n_workers',  type=int, default=None,
                        help='Parallel simulation workers (CPU). Default: auto = os.cpu_count(). '
                             'Sim generation is always CPU-bound regardless of GPU device.')
    parser.add_argument('--force',      action='store_true',
                        help='Continue even if existing chunks were generated under a '
                             'different config. Produces a dataset that mixes two '
                             'instrument setups — only use deliberately.')
    parser.add_argument('--max_outstanding_chunks', type=int, default=None,
                        help='Backpressure: pause generation once this many chunk files '
                             'exist in sim_chunks/, polling until a consumer deletes some '
                             '(train_sbi.py --delete_after_training) before writing more. '
                             'Bounds peak disk usage to roughly max_outstanding_chunks * '
                             'chunk_size * bytes/sim regardless of --n_sims, so --n_sims '
                             'can target a dataset larger than the disk quota could hold '
                             'all at once. Default: unbounded (no pausing).')
    parser.add_argument('--backpressure_poll_interval', type=float, default=30.0,
                        help='Seconds between checks while paused for backpressure '
                             '(default 30).')
    args = parser.parse_args(argv)

    # --------------------------------------------------
    # Config
    # Resolved to an ABSOLUTE path before the os.chdir() below, otherwise a
    # relative --config would stop resolving once we move into the outdir.
    # --------------------------------------------------
    config_path = os.path.abspath(resolve_config_path(args.config))
    cfg         = get_config(config_path)

    # --------------------------------------------------
    # Device detection (info only) + resolve n_workers
    # Sim generation is always CPU — n_workers is independent of GPU device.
    # --------------------------------------------------
    device     = get_device()
    device_str = str(device)
    cpu_count  = os.cpu_count() or 1

    n_workers = args.n_workers if args.n_workers is not None else cpu_count

    print(cfg.summary())
    print("=" * 62)
    print(f"  Device detected : {device_str.upper()}")
    print(f"  CPU cores       : {cpu_count}")
    print(f"  Sim workers     : {n_workers}  ({'parallel' if n_workers > 1 else 'serial'})")
    print(f"  n_sims          : {args.n_sims}")
    print(f"  chunk_size      : {args.chunk_size}")
    print("=" * 62)

    # --------------------------------------------------
    # Setup output directory
    # --------------------------------------------------
    outdir = args.outdir or f"sims_{cfg.name}_{args.n_sims}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    print("Creating dataset in:", os.getcwd())
    Path("sim_chunks").mkdir(exist_ok=True)

    # --------------------------------------------------
    # Continue chunk numbering if existing chunks present (resume scenario)
    #
    # Guard against config drift first: resuming with an edited YAML would
    # silently produce a dataset that is half one instrument and half another,
    # and nothing downstream could detect it.
    #
    # counting existing files (len(glob(...))) is NOT safe once
    # --delete_after_training has been in use on the consumer side: chunks
    # that were already generated and fed to a completed training round get
    # deleted, so the file count would undercount how many sims were ever
    # actually generated and hand out chunk_ids that were already used —
    # colliding with rounds training has already absorbed. sim_chunks/
    # .next_chunk_id is written atomically alongside every chunk (and never
    # deleted) specifically to survive that, and is trusted over the file
    # count whenever it exists.
    # --------------------------------------------------
    _existing = sorted(glob.glob("sim_chunks/sims_chunk_*.pt"))
    _progress_file = "sim_chunks/.next_chunk_id"
    if os.path.isfile(_progress_file):
        with open(_progress_file) as f:
            chunk_id = int(f.read().strip())
        print(f"Found progress marker — new chunks will start at index {chunk_id} "
              f"({len(_existing)} chunk files currently present; some earlier ones "
              f"may already have been deleted by a consumer).")
    else:
        chunk_id = len(_existing)
        if chunk_id > 0:
            print(f"Found {chunk_id} existing chunks — new chunks will start at index {chunk_id}.")
    if chunk_id > 0:
        if os.path.isfile("metadata.pt"):
            _old = torch.load("metadata.pt", map_location="cpu", weights_only=False)
            check_fingerprint(cfg, _old.get("config_fingerprint"),
                              "the existing chunks in this directory",
                              force=args.force)
            del _old
        else:
            print("WARNING: existing chunks but no metadata.pt — cannot verify "
                  "they came from this config.")

    def _update_progress_marker(next_id):
        tmp = _progress_file + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(next_id))
        os.replace(tmp, _progress_file)

    # --------------------------------------------------
    # --n_sims is always the WHOLE dataset's target size, identically on a
    # fresh start and a resume -- never "how many more to generate this
    # invocation". This generation loop only needs to produce whatever's
    # left: n_sims minus what chunk_id already accounts for.
    #
    # Previously this invocation's loop ran for the full args.n_sims
    # regardless of chunk_id, so resuming at chunk_id=572 with the same
    # --n_sims 30000000 used at the original launch generated 1000 MORE
    # chunks on top of the 572 already done -- overshooting to 1572 chunks
    # (47.16M sims) instead of the intended 1000 chunks (30M sims), and
    # train_sbi.py's round count followed that wrong total too. Caught
    # 2026-10-13 after only 2 stray chunks were made.
    # --------------------------------------------------
    n_to_generate = max(0, args.n_sims - chunk_id * args.chunk_size)
    if chunk_id > 0:
        print(f"{chunk_id * args.chunk_size:,} sims already accounted for by "
              f"existing chunks/marker — generating the remaining "
              f"{n_to_generate:,} of {args.n_sims:,} total this run.")
    if n_to_generate == 0:
        print("Nothing left to generate — already at or past the target n_sims.")
        raise SystemExit(0)

    # --------------------------------------------------
    # Prior — built from the SAME config the simulator samples from.
    # This is the single declaration point. Do not reintroduce literals here.
    # --------------------------------------------------
    prior = BoxUniform(
        low  = torch.from_numpy(cfg.prior_low),
        high = torch.from_numpy(cfg.prior_high),
    )

    # Atomic for the same reason chunks are (_save_chunk_atomic below):
    # train_sbi.py --wait_for_chunks can start polling for this file before
    # generation is even fully under way, and a reader that opens it mid-write
    # would get a truncated pickle rather than waiting for the real thing.
    _metadata_tmp = "metadata.pt.tmp"
    torch.save({
        "prior":              prior,
        "config_path":        config_path,
        "config_yaml":        cfg.text,          # verbatim — keeps this directory
                                                 # self-describing after a move
        "config_fingerprint": cfg.fingerprint,
        "param_names":        cfg.param_names,   # theta ordering
        "param_labels":       cfg.param_labels,
        "log_params":         [p.name for p in cfg.inferred if p.log],
        "n_chan":             cfg.n_chan,
        "n_bins":             cfg.n_bins,
        "x_dim":              cfg.x_dim,
        # The generation PLAN, written before any simulation runs, so
        # train_sbi.py --wait_for_chunks knows how many chunks to expect and
        # when generation should be complete. On a resume these count the
        # pre-existing chunks too, so the total is always the whole dataset.
        "n_sims":             args.n_sims,
        "chunk_size":         args.chunk_size,
        "expected_chunks":    math.ceil(args.n_sims / args.chunk_size),
        "chunks_before_run":  chunk_id,
    }, _metadata_tmp)
    os.replace(_metadata_tmp, "metadata.pt")

    print(f"Prior ({len(cfg.param_names)} params, from {os.path.basename(config_path)}):")
    for n, lo, hi in zip(cfg.param_names, cfg.prior_low, cfg.prior_high):
        print(f"    {n:<16} U[{lo:>10.4f}, {hi:>10.4f}]")

    theta_chunk = []
    x_chunk     = []
    count       = 0
    print(f"\nGenerating {n_to_generate} simulated pulsar portraits...")

    def _save_chunk_atomic(theta_list, x_list, cid):
        """Write one chunk so a concurrent reader never sees a partial file.

        torch.save() streams into the destination, so a reader that opens the
        file mid-write gets a truncated pickle — a crash if it is lucky, garbage
        if it is not. Writing to a temporary name and then os.replace()-ing it
        makes the appearance of the final name atomic on any POSIX filesystem,
        so a poller sees the chunk either absent or complete, never in between.

        This matters for train_sbi.py --wait_for_chunks, which polls this
        directory while generation is still running. It also matters WITHOUT
        streaming: a generation job killed mid-write would otherwise leave a
        truncated chunk that a later training run would load as real data.
        """
        final = f"sim_chunks/sims_chunk_{cid:04d}.pt"
        tmp   = f"{final}.tmp"
        torch.save(
            {"theta": torch.tensor(theta_list, dtype=torch.float32),
             "x":     torch.tensor(x_list,     dtype=torch.float32)},
            tmp,
        )
        os.replace(tmp, final)   # atomic within one filesystem

    def _append_and_maybe_save(sim):
        nonlocal count, chunk_id
        # theta comes pre-ordered with the log transform already applied
        # (cfg.theta_vector), so no parameter name appears here at all.
        theta_chunk.append(sim['theta'])
        x_chunk.append(np.concatenate([
            sim['portrait'].flatten(),
            sim['flux_per_channel'],
            sim['centroids'],
            # mean_profile — deliberately excluded: frequency-averaging bakes the
            # spectral indices into the profile shape, contaminating ToA.
            # centroid_residuals — removed with the scatter_branch.
        ]))
        count += 1
        if count % args.chunk_size == 0:
            _save_chunk_atomic(theta_chunk, x_chunk, chunk_id)
            chunk_id += 1
            _update_progress_marker(chunk_id)   # AFTER the chunk write, so a
                                                 # crash mid-write never advances
                                                 # the marker past a chunk that
                                                 # doesn't actually exist yet
            print(f"Saved chunk {chunk_id - 1}  ({count} / {n_to_generate})")
            theta_chunk.clear()
            x_chunk.clear()

    def _wait_for_backpressure():
        """Block until sim_chunks/ has room for another chunk.

        Workers in the Pool below pull tasks eagerly, independent of how fast
        the main process consumes results — pausing mid-consumption of a
        single giant imap_unordered() call would NOT stop them computing,
        just make already-finished results pile up in memory unboundedly.
        So generation is submitted in chunk_size-sized batches instead (see
        below): this check runs strictly BETWEEN batches, when no worker has
        outstanding work, which is the only point a pause is actually safe.
        """
        if args.max_outstanding_chunks is None:
            return
        while True:
            existing = len(glob.glob("sim_chunks/sims_chunk_*.pt"))
            if existing < args.max_outstanding_chunks:
                return
            print(f"  backpressure: {existing} chunks outstanding (limit "
                  f"{args.max_outstanding_chunks}) — waiting for a consumer "
                  f"to delete some before generating more ...")
            time.sleep(args.backpressure_poll_interval)

    # --------------------------------------------------
    # Generate simulations
    # --------------------------------------------------
    if n_workers > 1:
        # Fixed seed (42) regardless of fresh-start or resume, so the FULL
        # eventual dataset's seed sequence is reproducible end to end. A
        # resume must slice into this same deterministic stream at the
        # chunk_id offset, not restart it from draw 0 -- otherwise it
        # redraws the same seeds already used for chunks 0..chunk_id-1 and
        # silently duplicates that simulated data under new chunk numbers.
        rng       = np.random.default_rng(42)
        already   = chunk_id * args.chunk_size
        all_seeds = rng.integers(0, 2**31, size=already + n_to_generate)
        seeds     = all_seeds[already:].tolist()
        worker = partial(_sim_worker, config_path=config_path)
        with mp.Pool(processes=n_workers) as pool:
            for batch_start in range(0, n_to_generate, args.chunk_size):
                _wait_for_backpressure()
                batch_seeds = seeds[batch_start:batch_start + args.chunk_size]
                for sim in pool.imap_unordered(worker, batch_seeds, chunksize=50):
                    _append_and_maybe_save(sim)
    else:
        # NOTE: sim_generator's internal seeding is not offset-adjusted for
        # chunk_id the way the multi-worker path above now is -- a resume
        # through this single-worker path may still redraw already-used
        # sims. Not fixed here since no current sbatch script uses
        # n_workers=1; flagging so it isn't assumed safe if that changes.
        for sim in sim_generator(n_to_generate, config_path=config_path):
            _wait_for_backpressure()
            _append_and_maybe_save(sim)

    # Save remainder
    if theta_chunk:
        _save_chunk_atomic(theta_chunk, x_chunk, chunk_id)
        print(f"Saved final chunk {chunk_id}  ({count} / {n_to_generate})")
        _update_progress_marker(chunk_id + 1)

    print(f"Dataset generation complete.  ({cfg.name} @ {cfg.obs_name}, "
          f"fingerprint {cfg.fingerprint})")


if __name__ == '__main__':
    main()
