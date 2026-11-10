import time
import json
from .efficiency import ChunkStats, RoundStats, compute_round_efficiency, get_hardware_profile
import torch
import numpy as np
import glob
import argparse
import gc
import math
import os
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from .pulsarDMvary import (get_device, build_encoder, encoder_param_count,
                          ENCODERS)
from .pulsar_config import get_config, check_fingerprint, recorded_config_path

# --------------------------------------------------
# Arguments
# --------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str,   required=True)
    parser.add_argument('--config',     type=str,   default=None,
                        help="Path to the YAML config. Default: the config recorded in "
                             "<dataset>/metadata.pt, which is almost always what you want "
                             "— it cannot drift from the dataset.")
    parser.add_argument('--encoder',    type=str,   default='channelset',
                        choices=sorted(ENCODERS),
                        help="Embedding network. 'channelset' = ChannelSetEncoder "
                             "(per-channel tokens + self-attention, ~309k params). "
                             "'multibranch' = fixed 2D CNN (~139k params).")
    parser.add_argument('--force',      action='store_true',
                        help="Train even if the config does not match the dataset's "
                             "fingerprint. Only use deliberately.")
    parser.add_argument('--wait_for_chunks', action='store_true',
                        help="Stream: start training on the chunks that exist and WAIT for "
                             "the rest as sim_maker.py writes them, instead of requiring "
                             "the dataset to be complete. Lets the CPU generation job and "
                             "this GPU job overlap. Chunk mode only — single_pass needs the "
                             "whole dataset at once. Requires 'expected_chunks' in "
                             "metadata.pt, which sim_maker.py writes before generating.")
    parser.add_argument('--poll_interval', type=float, default=30.0,
                        help="Seconds between checks for new chunks with "
                             "--wait_for_chunks (default 30).")
    parser.add_argument('--stall_timeout', type=float, default=1800.0,
                        help="Abort if no new chunk appears for this many seconds while "
                             "waiting (default 1800 = 30 min). This is the ONLY way to "
                             "detect a dead generation job: it runs on a different Slurm "
                             "cluster, so squeue here cannot see it.")
    parser.add_argument('--epochs',     type=int,   default=None,
                        help="Max epochs. Auto: 500 (CUDA / single_pass), 40 (MPS / chunk). "
                             "Explicit value always overrides auto.")
    parser.add_argument('--batch_size', type=int,   default=None,
                        help="Training batch size. Auto: 4096 (CUDA), 256 (MPS/CPU).")
    parser.add_argument('--stop_after', type=int,   default=None,
                        help="Early-stopping patience in epochs. Auto: 30 (CUDA), 20 (MPS/CPU).")
    parser.add_argument('--mode',       type=str,   default='auto',
                        choices=['auto', 'single_pass', 'chunk'],
                        help="'auto' picks best mode for detected device. "
                             "'single_pass' loads all data into RAM and trains once (no catastrophic forgetting). "
                             "'chunk' is the original sequential loop (MPS/CPU default).")
    parser.add_argument('--resume',     type=str,   default=None,
                        help="Path to a previous posterior.pt to warm-start weights from. "
                             "Always starts back at round 0 — for continuing an interrupted "
                             "chunk-mode run without repeating already-completed rounds, use "
                             "--resume_checkpoint instead.")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to a checkpoint.pt saved mid-run by chunk-mode training "
                             "(written after every round — see round loop). Resumes weights "
                             "AND skips rounds already completed, unlike --resume. Chunk mode "
                             "only; mutually exclusive with --resume.")
    parser.add_argument('--delete_after_training', action='store_true',
                        help="Delete each round's chunk file(s) immediately after that "
                             "round's checkpoint is successfully saved. Lets --n_sims target "
                             "a dataset larger than the disk quota could hold all at once, "
                             "paired with sim_maker.py --max_outstanding_chunks. Safe because "
                             "chunk mode with group_size=1 (or any group_size) touches each "
                             "chunk exactly once and never revisits it, and simul_valid.py / "
                             "pp_tests.py generate their own fresh test simulations rather "
                             "than reading these back. Chunk mode only. Deletion happens AFTER "
                             "the checkpoint write, never before, so a crash mid-round never "
                             "leaves you needing a chunk that is already gone.")
    parser.add_argument('--lr',         type=float, default=2e-4,
                        help="Learning rate (default 2e-4; use ~1e-4 when resuming to reduce forgetting).")
    parser.add_argument('--training_log', type=str, default=None,
                        help="Path to training.log. Defaults to <dataset>/training.log. "
                             "If found, validation losses are listed and plotted at the end.")
    parser.add_argument('--group_size', type=int,   default=1,
                        help="Number of chunk files to concatenate per training round (chunk mode only). "
                             "Default 1 = original file-by-file. Use e.g. 10 to train on 100k sims "
                             "at a time on GPU, keeping the GPU saturated without a full single_pass OOM.")
    parser.add_argument('--seed',       type=int,   default=None,
                        help="Random seed for reproducibility. If not set, a random seed is drawn and logged.")
    args = parser.parse_args(argv)

    # --------------------------------------------------
    # Reproducibility seed
    # --------------------------------------------------
    seed = args.seed if args.seed is not None else torch.seed() & 0xFFFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Training seed: {seed}")

    # --------------------------------------------------
    # Device detection → resolve training hyper-params
    # --------------------------------------------------
    device     = get_device()
    device_str = str(device)   # 'cuda', 'mps', or 'cpu'
    sbi_device = device_str if device_str in ('cuda', 'cpu') else 'cpu'  # SBI doesn't support mps

    # Fill in any None args based on device
    if device_str == 'cuda':
        batch_size  = args.batch_size  if args.batch_size  is not None else 4096
        epochs      = args.epochs      if args.epochs      is not None else 500
        stop_after  = args.stop_after  if args.stop_after  is not None else 30
        use_single_pass = (args.mode == 'single_pass')    # auto → False (chunk), single_pass → True, chunk → False
    else:  # mps or cpu
        batch_size  = args.batch_size  if args.batch_size  is not None else 256
        epochs      = args.epochs      if args.epochs      is not None else 40
        stop_after  = args.stop_after  if args.stop_after  is not None else 20
        use_single_pass = (args.mode == 'single_pass')    # auto → False, single_pass → True, chunk → False

    # --------------------------------------------------
    # Load metadata, then resolve the config
    #
    # The config defaults to the one recorded in metadata.pt when the dataset was
    # generated, so the common case needs no --config flag and CANNOT drift from
    # the data. An explicit --config is checked against the dataset's fingerprint.
    #
    # With --wait_for_chunks this file may not exist yet: sim_maker.py and this
    # process are submitted to independent Slurm clusters with no ordering
    # guarantee between them, so the GPU job can easily start before sim_maker.py
    # has even created the dataset directory. Wait for it exactly like a chunk,
    # rather than letting torch.load crash on a race that --wait_for_chunks was
    # supposed to handle.
    # --------------------------------------------------
    metadata_path = f"{args.dataset}/metadata.pt"
    if args.wait_for_chunks and not os.path.isfile(metadata_path):
        print(f"Streaming mode: waiting for {metadata_path} "
              f"(sim_maker.py writes this before generating anything) ...")
        t_wait_start = time.time()
        while not os.path.isfile(metadata_path):
            time.sleep(args.poll_interval)
            stalled = time.time() - t_wait_start
            if stalled > args.stall_timeout:
                raise SystemExit(
                    f"ERROR: {metadata_path} not found after {stalled/60:.1f} min "
                    f"(--stall_timeout {args.stall_timeout/60:.1f} min). Either "
                    f"sim_maker.py has not started, --dataset points somewhere "
                    f"wrong, or it already failed before writing metadata.pt."
                )
        print(f"  found after {time.time() - t_wait_start:.1f}s")

    meta  = torch.load(metadata_path, map_location=sbi_device, weights_only=False)
    prior = meta["prior"]

    config_path = recorded_config_path(meta, args.dataset, args.config)
    cfg = get_config(config_path)
    check_fingerprint(cfg, meta.get("config_fingerprint"),
                      f"dataset {args.dataset}", force=args.force)

    # --------------------------------------------------
    # Build network — shaped by the config, so the encoder can never disagree
    # with the feature layout the dataset actually has.
    # --------------------------------------------------
    encoder = build_encoder(args.encoder, cfg).float().to(device)

    print("=" * 62)
    print(f"  Device detected : {device_str.upper()}")
    print(f"  Config          : {os.path.basename(cfg.path)}  "
          f"(fingerprint {cfg.fingerprint})")
    print(f"  Pulsar / site   : {cfg.name} @ {cfg.obs_name}")
    print(f"  Instrument      : {cfg.n_chan} ch x {cfg.n_bins} bins  "
          f"({cfg.freq_low_mhz:.2f}-{cfg.freq_high_mhz:.2f} MHz)")
    print(f"  Encoder         : {args.encoder}  "
          f"({encoder_param_count(encoder):,} params)")
    print(f"  Inferred params : {len(cfg.param_names)}  ({', '.join(cfg.param_names)})")
    print(f"  Feature width   : {cfg.x_dim}")
    print(f"  Training mode   : {'single_pass' if use_single_pass else 'chunk'}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Max epochs      : {epochs}")
    print(f"  Stop after      : {stop_after} epochs without improvement")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Random seed     : {seed}")
    if args.resume:
        print(f"  Resuming from   : {args.resume}")
    if args.resume_checkpoint:
        print(f"  Resuming from checkpoint: {args.resume_checkpoint}")
    print("=" * 62)

    if args.resume and args.resume_checkpoint:
        raise SystemExit(
            "ERROR: --resume and --resume_checkpoint are mutually exclusive — "
            "--resume warm-starts from a completed posterior.pt and always begins "
            "at round 0; --resume_checkpoint continues a chunk-mode run from where "
            "it stopped, including which round to resume at. Pick one."
        )
    if args.resume_checkpoint and use_single_pass:
        raise SystemExit(
            "ERROR: --resume_checkpoint is chunk-mode only — single_pass has no "
            "rounds to resume partway through."
        )
    if args.delete_after_training and use_single_pass:
        raise SystemExit(
            "ERROR: --delete_after_training is chunk-mode only — single_pass loads "
            "every chunk before training even starts, so there is nothing left to "
            "delete round-by-round."
        )

    print("Using device:", device)

    # GPU efficiency (huge on clusters)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("medium")

    # --------------------------------------------------
    # Density estimator (encoder built above, alongside the config)
    # --------------------------------------------------
    neural_posterior = posterior_nn(
        model="nsf",
        embedding_net=encoder,
        hidden_features=128,
        num_components=5
    )

    # --------------------------------------------------
    # Chunk file list
    # --------------------------------------------------
    CHUNK_GLOB = f"{args.dataset}/sim_chunks/sims_chunk_*.pt"


    def chunk_path(i):
        return f"{args.dataset}/sim_chunks/sims_chunk_{i:04d}.pt"


    def present_chunks():
        """Chunks currently complete on disk.

    sim_maker.py writes to `<name>.tmp` and os.replace()s, so a name matching
    this glob is always a fully-formed file. The glob deliberately does not
    match `.tmp`.
    """
        return sorted(glob.glob(CHUNK_GLOB))


    chunk_files = present_chunks()
    print("Found", len(chunk_files), "chunks")

    # How many chunks the generation job intends to produce, recorded by
    # sim_maker.py in metadata.pt before it generates anything.
    expected_chunks = meta.get("expected_chunks")

    if args.wait_for_chunks:
        if expected_chunks is None:
            raise SystemExit(
                f"--wait_for_chunks needs 'expected_chunks' in "
                f"{args.dataset}/metadata.pt, which this dataset predates. Either "
                f"re-run sim_maker.py (it writes metadata.pt before generating, so "
                f"you can start it and then start training), or drop "
                f"--wait_for_chunks and train on what is already there."
            )
        print(f"Streaming mode: generation plans {expected_chunks} chunks "
              f"({meta.get('n_sims','?')} sims at chunk_size "
              f"{meta.get('chunk_size','?')}); {len(chunk_files)} present so far.")
    elif not chunk_files:
        raise SystemExit(f"No chunks in {args.dataset}/sim_chunks/ — run sim_maker.py first.")


    def wait_for_chunk_range(lo, hi):
        """Block until chunks [lo, hi) are all present. Returns their paths.

    Polls rather than using inotify so it works across the shared filesystem
    between two clusters, where the generating job is not visible to this one.
    That invisibility is also why a stall timeout is the only way to notice a
    dead generator: `squeue` on this cluster cannot see a job on the other, so
    "no new chunk for --stall_timeout seconds" stands in for "the producer
    died". Set it generously — a busy filesystem or a requeued job can stall
    legitimately for a while.
    """
        needed = [chunk_path(i) for i in range(lo, hi)]
        missing = [p for p in needed if not os.path.exists(p)]
        if not missing:
            return needed

        n_before = len(present_chunks())
        t_last_progress = time.time()
        waited = 0.0
        print(f"  waiting for chunks {lo}–{hi-1}: {len(missing)} of {hi-lo} not yet written")

        while missing:
            time.sleep(args.poll_interval)
            waited += args.poll_interval
            n_now = len(present_chunks())
            if n_now > n_before:
                # Progress anywhere in the directory resets the stall clock: the
                # producer is alive even if it has not reached our range yet.
                n_before = n_now
                t_last_progress = time.time()
            stalled = time.time() - t_last_progress
            if stalled > args.stall_timeout:
                raise SystemExit(
                    f"ERROR: no new chunk for {stalled/60:.1f} min (--stall_timeout "
                    f"{args.stall_timeout/60:.1f} min) while waiting for chunks "
                    f"{lo}–{hi-1}. {n_now}/{expected_chunks} chunks present. The "
                    f"generation job has most likely died — check it, then restart "
                    f"training (it will pick up every chunk already written)."
                )
            missing = [p for p in needed if not os.path.exists(p)]
            if waited % max(args.poll_interval, 60) < args.poll_interval or not missing:
                print(f"    {n_now}/{expected_chunks} chunks present, "
                      f"{len(missing)} still needed for this round "
                      f"({waited/60:.1f} min waited, stall clock {stalled/60:.1f} min)")
        print(f"  chunks {lo}–{hi-1} ready after {waited/60:.1f} min")
        return needed


    # --------------------------------------------------
    # Feature-width reconciliation
    #
    # Both widths are DERIVED from the config rather than hardcoded. The previous
    # code compared against literal 20600 / 20560, which are 40*512 + 40*k — values
    # that mean nothing under any other instrument, and worse, let any other width
    # pass through unchecked into append_simulations().
    # --------------------------------------------------
    X_DIM          = cfg.x_dim                    # portrait + flux + centroids
    X_DIM_LEGACY   = X_DIM + cfg.n_chan           # ... + centroid_residuals


    def reconcile_x(x):
        """Return `x` trimmed to the config's feature width, or abort."""
        got = x.shape[1]
        if got == X_DIM:
            return x
        if got == X_DIM_LEGACY:
            print(f"  Legacy chunks ({X_DIM_LEGACY} dims) — stripping trailing "
                  f"centroid_residuals ({cfg.n_chan} dims) to reach {X_DIM}.")
            return x[:, :X_DIM].contiguous()
        raise SystemExit(
            f"ERROR: feature width {got} matches neither the config's expected "
            f"{X_DIM} nor the legacy {X_DIM_LEGACY}.\n"
            f"  Config {os.path.basename(cfg.path)} describes {cfg.n_chan} ch x "
            f"{cfg.n_bins} bins -> {cfg.n_chan}*{cfg.n_bins} + 2*{cfg.n_chan} = {X_DIM}.\n"
            f"  This is the wrong config for this dataset, or the chunks were written "
            f"with a different feature layout."
        )

    round_stats = RoundStats(round_id=1)

    # Warm-start weights from a previous run if --resume is supplied
    trained_state_dict = None
    if args.resume:
        print(f"Resuming weights from: {args.resume}")
        _prev = torch.load(args.resume, map_location="cpu", weights_only=False)
        trained_state_dict = _prev.posterior_estimator.state_dict()
        del _prev
        print("Previous weights loaded — will warm-start before training.")

    # Continue an interrupted chunk-mode run: load weights AND skip rounds already
    # completed, so a resumed run doesn't retrain on early chunks twice while
    # later chunks are only ever seen once (which --resume alone would do, since
    # it always restarts the round loop at 0).
    start_round = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        _ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        trained_state_dict = _ckpt["state_dict"]
        start_round = _ckpt["round"] + 1
        print(f"  round {_ckpt['round']} was the last one completed — "
              f"continuing at round {start_round}")

    hardware = get_hardware_profile()

    # ==================================================
    # SINGLE-PASS training  (cuda auto / --mode single_pass)
    # ==================================================
    if use_single_pass and args.wait_for_chunks:
        raise SystemExit(
            "--wait_for_chunks is incompatible with --mode single_pass: single_pass "
            "hands the WHOLE dataset to sbi at once, so there is nothing to stream "
            "into. Use --mode chunk to overlap generation and training, or drop "
            "--wait_for_chunks and let generation finish first."
        )

    if use_single_pass:
        print("\nSingle-pass mode: loading all chunks into CPU RAM ...")

        # Determine total size from first and last chunks to pre-allocate without double-buffering.
        # torch.cat holds both the list AND the new tensor simultaneously (~2x RAM spike), which
        # OOMs on large datasets. Pre-allocating avoids this: peak = final tensor + one chunk.
        _first = torch.load(chunk_files[0],  map_location="cpu", weights_only=False)
        _last  = torch.load(chunk_files[-1], map_location="cpu", weights_only=False)
        theta_dim = _first["theta"].shape[1]
        x_dim     = _first["x"].shape[1]
        n_per_chunk = _first["theta"].shape[0]
        n_total = (len(chunk_files) - 1) * n_per_chunk + _last["theta"].shape[0]
        del _first, _last

        theta_all = torch.empty(n_total, theta_dim, dtype=torch.float32)
        x_all     = torch.empty(n_total, x_dim,     dtype=torch.float32)

        idx = 0
        for i, fname in enumerate(chunk_files):
            data = torch.load(fname, map_location="cpu", weights_only=False)
            n = data["theta"].shape[0]
            theta_all[idx:idx+n] = data["theta"]
            x_all[idx:idx+n]     = data["x"]
            del data
            idx += n
            if (i + 1) % 10 == 0:
                print(f"  {i + 1} / {len(chunk_files)} chunks loaded ...")

        x_all = reconcile_x(x_all)

        if theta_all.shape[1] != len(cfg.param_names):
            raise SystemExit(
                f"ERROR: theta has {theta_all.shape[1]} columns but config "
                f"{os.path.basename(cfg.path)} declares {len(cfg.param_names)} "
                f"inferred parameters ({', '.join(cfg.param_names)}).\n"
                f"  The inferred set or its order changed after this dataset was made."
            )

        theta_gb = theta_all.element_size() * theta_all.nelement() / 1e9
        x_gb     = x_all.element_size()     * x_all.nelement()     / 1e9
        print(f"Loaded {n_total:,} simulations  (theta {theta_gb:.2f} GB + x {x_gb:.2f} GB in CPU RAM)")

        inference = SNPE(prior=prior, density_estimator=neural_posterior, device=sbi_device)
        inference.append_simulations(theta_all, x_all)   # kept on CPU; DataLoader moves batches to device
        del theta_all, x_all   # release our references; SBI holds its own copy if needed
        gc.collect()

        if trained_state_dict is not None:
            # Initialise network structure with a 0-epoch pass, then inject weights
            inference.train(max_num_epochs=0, force_first_round_loss=True)
            inference._neural_net.load_state_dict(trained_state_dict)

        t0 = time.time()
        density_estimator = inference.train(
            training_batch_size=batch_size,
            learning_rate=args.lr,
            max_num_epochs=epochs,
            validation_fraction=0.1,
            stop_after_epochs=stop_after,
            show_train_summary=True,
            force_first_round_loss=True,
            clip_max_norm=5.0,
        )
        train_time = time.time() - t0

        val_loss      = inference._summary["best_validation_loss"]
        epochs_trained = inference._summary["epochs_trained"]
        if isinstance(val_loss,       list): val_loss       = val_loss[-1]
        if isinstance(epochs_trained, list): epochs_trained = epochs_trained[-1]

        chunk_stats = ChunkStats(
            chunk_id=0,
            num_simulations=n_total,
            epochs_trained=epochs_trained,
            best_val_loss=val_loss,
            time_seconds=train_time,
            cumulative_simulations=n_total,
        )
        round_stats.add_chunk(chunk_stats)

    # ==================================================
    # CHUNK-BY-CHUNK training  (mps/cpu auto / --mode chunk)
    # ==================================================
    else:
        group_size = max(1, args.group_size)

        if args.wait_for_chunks:
            # Plan the rounds over every chunk generation INTENDS to write, not
            # just those already on disk, and block per round until its files
            # appear. Each round then trains exactly as it always did.
            n_total_chunks = expected_chunks
            groups = None          # resolved lazily, one round at a time
            print(f"\nStreaming {n_total_chunks} chunks in "
                  f"{math.ceil(n_total_chunks / group_size)} rounds of {group_size}; "
                  f"poll {args.poll_interval:g}s, stall timeout "
                  f"{args.stall_timeout/60:.0f} min")
        else:
            n_total_chunks = len(chunk_files)
            groups = [chunk_files[i:i+group_size]
                      for i in range(0, n_total_chunks, group_size)]

        n_groups = math.ceil(n_total_chunks / group_size)
        cumulative_sims = 0

        if start_round >= n_groups:
            raise SystemExit(
                f"ERROR: checkpoint says round {start_round - 1} was the last completed, "
                f"but this run only has {n_groups} rounds total — nothing left to do."
            )

        for g in range(start_round, n_groups):
            lo, hi = g * group_size, min((g + 1) * group_size, n_total_chunks)
            group = (wait_for_chunk_range(lo, hi) if args.wait_for_chunks
                     else groups[g])

            group_start_time = time.time()
            print(f"\n========== GROUP {g+1}/{n_groups}  ({len(group)} chunk files) ==========")

            # Load and concatenate all files in this group
            theta_parts, x_parts = [], []
            group_sims = 0
            for fname in group:
                data = torch.load(fname, map_location="cpu", weights_only=False)
                theta_parts.append(data["theta"])
                x_parts.append(data["x"])
                group_sims += data["theta"].shape[0]
                del data
            theta = torch.cat(theta_parts, dim=0)
            x     = torch.cat(x_parts,     dim=0)
            del theta_parts, x_parts
            gc.collect()
            x = reconcile_x(x)
            print(f"  Loaded {group_sims:,} simulations")

            # Fresh SNPE object each group (sbi requirement)
            inference = SNPE(prior=prior, density_estimator=neural_posterior, device=sbi_device)
            inference.append_simulations(theta.to(device), x.to(device))
            del theta, x
            gc.collect()

            if trained_state_dict is not None:
                density_estimator = inference.train(
                    max_num_epochs=0,
                    force_first_round_loss=True
                )
                inference._neural_net.load_state_dict(trained_state_dict)

            density_estimator = inference.train(
                training_batch_size=batch_size,
                learning_rate=args.lr,
                max_num_epochs=epochs,
                validation_fraction=0.1,
                stop_after_epochs=stop_after,
                show_train_summary=True,
                force_first_round_loss=True,
                clip_max_norm=5.0,
            )

            trained_state_dict = density_estimator.state_dict()

            # Checkpoint every round: cheap (a few MB) next to a round that takes
            # minutes, and means a crash/timeout/preemption loses at most the
            # round in progress rather than everything since round 0. Atomic
            # write — same temp-then-os.replace pattern as chunks and metadata.pt
            # — so a checkpoint a resumed run loads is always complete.
            torch.save({"round": g, "state_dict": trained_state_dict, "seed": seed},
                       f"{args.dataset}/checkpoint.pt.tmp")
            os.replace(f"{args.dataset}/checkpoint.pt.tmp", f"{args.dataset}/checkpoint.pt")

            # Only after the checkpoint above is confirmed on disk: this round's
            # chunks are now fully absorbed into trained_state_dict and will never
            # be read again (group_size chunks are consumed exactly once, never
            # revisited), so deleting them is safe and lets sim_maker.py's
            # --max_outstanding_chunks backpressure keep generating further ahead.
            if args.delete_after_training:
                for fname in group:
                    os.remove(fname)
                print(f"  Deleted {len(group)} chunk file(s) for this round.")

            group_time       = time.time() - group_start_time
            val_loss         = inference._summary["best_validation_loss"]
            cumulative_sims += group_sims
            epochs_trained = inference._summary["epochs_trained"]

            if isinstance(epochs_trained, list): epochs_trained = epochs_trained[-1]
            if isinstance(val_loss,       list): val_loss       = val_loss[-1]

            chunk_stats = ChunkStats(
                chunk_id=g,
                num_simulations=group_sims,
                epochs_trained=epochs_trained,
                best_val_loss=val_loss,
                time_seconds=group_time,
                cumulative_simulations=cumulative_sims,
            )
            round_stats.add_chunk(chunk_stats)

            if g + 1 < n_groups:
                del inference, density_estimator
                torch.cuda.empty_cache()
                gc.collect()

    # ==================================================
    # Efficiency log + save posterior
    # ==================================================
    score = compute_round_efficiency(round_stats, hardware)

    with open("efficiency_log.json", "a") as f:
        json.dump(score, f, indent=2)
        f.write("\n")

    print(f"Total time         : {score['cumulative']['total_time_seconds']}s")
    print(f"Final val loss     : {score['cumulative']['final_val_loss']}")
    print(f"Loss improvement   : {score['cumulative']['loss_improvement']}")
    print(f"Transfer efficiency: {score['efficiency']['transfer_efficiency']}")

    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, f"{args.dataset}/posterior.pt")
    print("Posterior saved.")

    # Sidecar describing what this posterior means. Kept SEPARATE from posterior.pt
    # rather than wrapping it in a dict, so --resume (which loads posterior.pt and
    # reads .posterior_estimator) and any posterior.pt already on disk keep working.
    #
    # Downstream scripts (DM_sampling.py, pp_tests.py, PPC.py, timeseries.py) should
    # read parameter names, labels and which params are logged from HERE instead of
    # hardcoding them — that is the remaining half of the refactor.
    torch.save({
        "config_path":        cfg.path,
        "config_yaml":        cfg.text,
        "config_fingerprint": cfg.fingerprint,
        "param_names":        cfg.param_names,
        "param_labels":       cfg.param_labels,
        "log_params":         [p.name for p in cfg.inferred if p.log],
        "log_param_indices":  [i for i, p in enumerate(cfg.inferred) if p.log],
        "encoder":            args.encoder,
        "encoder_params":     encoder_param_count(encoder),
        "n_chan":             cfg.n_chan,
        "n_bins":             cfg.n_bins,
        "x_dim":              cfg.x_dim,
        "prior_low":          cfg.prior_low,
        "prior_high":         cfg.prior_high,
        "seed":               seed,
    }, f"{args.dataset}/posterior_meta.pt")
    print(f"Posterior metadata saved -> {args.dataset}/posterior_meta.pt")

    print("The prior used for training is:", prior)

    # --------------------------------------------------
    # Validation loss summary + multi-run comparison
    # --------------------------------------------------
    import sys as _sys
    _log_arg = f" --training_log {args.training_log}" if args.training_log else ""
    _ret = os.system(
        f"{_sys.executable} -u -m p2sbi.val_loss_plot --dataset {args.dataset}{_log_arg}"
    )
    if _ret != 0:
        print("WARNING: val_loss_plot.py exited with an error — check output above.")


if __name__ == "__main__":
    main()
