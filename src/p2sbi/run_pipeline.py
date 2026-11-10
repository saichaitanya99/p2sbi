import argparse
import json
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
import glob
import torch


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def run_step(name, command, dataset_dir, pipeline_log):
    """
    Runs a pipeline stage using os.system with live output and logfile.
    Aborts pipeline if step fails.
    """

    step_log = os.path.join(dataset_dir, f"{name.lower().replace(' ', '_')}.log")

    print(f"\n===== {name} STARTED =====")
    print(f"Log file: {step_log}")

    if name == "TRAINING":
        print("NOTE: Training may take a long time. Monitor the log file.")

    start = time.time()

    # Write to pipeline master log
    with open(pipeline_log, "a") as log:
        log.write(f"\n[{datetime.now()}] START {name}\n")
        log.write("COMMAND: " + command + "\n")

    # bash pipefail is CRITICAL so tee does not hide crashes
    full_cmd = f'bash -o pipefail -c "{command} 2>&1 | tee -a {step_log}"'

    ret = os.system(full_cmd)

    elapsed = (time.time() - start) / 60.0

    if ret != 0:
        print(f"\n===== {name} FAILED after {elapsed:.2f} min =====")
        print(f"Check log: {step_log}")
        exit(1)

    # Move tensorboard logs automatically
    if name == "TRAINING":
        sbi_logs = "sbi-logs"
        if os.path.exists(sbi_logs):
            shutil.move(sbi_logs, os.path.join(dataset_dir, sbi_logs))
        if os.path.exists("efficiency_log.json"):
            shutil.move("efficiency_log.json", os.path.join(dataset_dir, "efficiency_log.json"))
    

    with open(pipeline_log, "a") as log:
        log.write(f"[{datetime.now()}] END {name} ({elapsed:.2f} min)\n")

    print(f"===== {name} FINISHED ({elapsed:.2f} min) =====\n")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Full SBI pulsar pipeline controller"
    )

    # Pipeline stages
    parser.add_argument('--make-dataset', action='store_true', help="Run dataset generation stage")
    parser.add_argument('--train', action='store_true', help="Run training stage")
    parser.add_argument('--make-test', action='store_true', help="Run test simulation stage")
    parser.add_argument('--sample', action='store_true', help="Run sampling stage")
    parser.add_argument('--PPC', action='store_true', help="Run posterior predictive checks stage")
    parser.add_argument('--pp-test', action='store_true', help="Run PP calibration tests stage (can run standalone if posterior exists)")
    parser.add_argument('--simul-valid', action='store_true', dest='simul_valid',
                        help="Run simulation validation (true-vs-recovered grid) after training")
    parser.add_argument('--val-loss-plot', action='store_true', dest='val_loss_plot',
                        help="Plot validation loss curves across all SBI_prods_* runs (no training needed)")
    parser.add_argument('--all', action='store_true', help="Run all stages in order (overrides individual stage flags)")

    # Config
    parser.add_argument('--config', type=str, default=None,
                        help="Path to the YAML config describing the pulsar, telescope, "
                             "priors and profile components. Threaded through every "
                             "stage. Required unless $SBI_PULSAR_CONFIG is set.")
    parser.add_argument('--encoder', type=str, default='channelset',
                        choices=['channelset', 'multibranch'],
                        help="Embedding network for training (default: channelset)")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset directory (default: SBI_prods_{n_sims})")
    parser.add_argument('--n_sims', type=int, default=10000, help="Number of simulations to generate for training (default: 10000)")
    parser.add_argument('--chunk_size', type=int, default=10000, help="Simulations per chunk file (default: 10000)")
    parser.add_argument('--n_workers', type=int, default=None,
                        help="Parallel simulation workers for dataset generation. "
                             "Passed straight to sim_maker.py; omitted means its own "
                             "default of os.cpu_count(). On SLURM set this to "
                             "$SLURM_CPUS_PER_TASK, since os.cpu_count() reports the "
                             "whole node rather than your cgroup allocation.")
    parser.add_argument('--max_outstanding_chunks', type=int, default=None,
                        help="Backpressure for --make-dataset: pause generation once this "
                             "many chunk files exist, polling until --train --"
                             "delete_after_training deletes some. Lets --n_sims target a "
                             "dataset larger than the disk quota could hold all at once. "
                             "Passed straight to sim_maker.py. Default: unbounded.")
    parser.add_argument('--delete_after_training', action='store_true',
                        help="For --train: delete each round's chunk file(s) once that "
                             "round's checkpoint is safely saved. Pair with --make-dataset "
                             "--max_outstanding_chunks on the generation side. Chunk mode "
                             "only. Passed straight to train_sbi.py.")
    parser.add_argument('--posterior', type=str, default=None, help="Path to posterior tensor (default: {dataset_dir}/posterior.pt)")
    parser.add_argument('--test_sim', type=str, default=None, help="Path to test simulation tensor (default: {dataset_dir}/sim*.pt)")
    parser.add_argument('--epochs', type=int, default=40, help="Number of training epochs per chunk(default: 40)")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Training batch size, passed to train_sbi.py. Default: "
                             "train_sbi.py's own auto value (4096 CUDA / 256 CPU/MPS) — "
                             "can OOM on memory-constrained GPUs. Verify with "
                             "tools/probe_vram.py before relying on the default.")
    parser.add_argument('--sims_per_round', type=int, default=None,
                        help="Target sims resident on GPU per training round (chunk mode). "
                             "Converted to train_sbi.py's --group_size as "
                             "sims_per_round // chunk_size. Default: 100,000 (historical "
                             "value) — verify with tools/probe_vram.py on a new GPU/config.")
    parser.add_argument('--wait_for_chunks', action='store_true',
                        help="Stream: start --train on whatever chunks exist and wait for "
                             "the rest as a concurrently-running --make-dataset writes "
                             "them, instead of requiring the dataset to be complete first. "
                             "Passed straight to train_sbi.py. Submit the --make-dataset "
                             "and --train invocations as separate, independent jobs (no "
                             "--dependency) for this to actually overlap them.")
    parser.add_argument('--poll_interval', type=float, default=30.0,
                        help="Seconds between chunk-availability checks with "
                             "--wait_for_chunks (default 30).")
    parser.add_argument('--stall_timeout', type=float, default=1800.0,
                        help="Abort --train if no new chunk appears for this many seconds "
                             "while streaming (default 1800 = 30 min).")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to a checkpoint.pt saved mid-run by chunk-mode training "
                             "(written every round). Passed straight to train_sbi.py: "
                             "resumes weights AND skips already-completed rounds — unlike "
                             "--resume above, which is a different, unrelated flag (archives "
                             "the dataset dir and starts new sims at round 0).")
    parser.add_argument('--n_test_cases', type=int, default=100, help="Number of PP test cases to run (default: 100)")
    parser.add_argument('--n_valid', type=int, default=1000, help="Number of test simulations for simul_valid.py (default: 1000)")
    parser.add_argument('--n_valid_samples', type=int, default=5000, help="Posterior samples per test case in simul_valid.py (default: 5000)")
    parser.add_argument('--seed',   type=int,  default=None,
                        help="Random seed passed to train_sbi.py for reproducibility (default: random).")
    parser.add_argument('--resume', action='store_true',
                        help="Resume from an existing --dataset directory. Archives its current "
                             "contents into old_train/, generates --n_sims new simulations, and "
                             "fine-tunes only on those (warm-starting from old_train/posterior.pt). "
                             "Unrelated to resuming an interrupted --make-dataset run -- that "
                             "happens automatically (see --fresh_start below).")
    parser.add_argument('--fresh_start', action='store_true',
                        help="DESTRUCTIVE: wipe --dataset's directory before generating, instead "
                             "of the default safe behaviour of leaving existing contents alone "
                             "and letting sim_maker.py's own fingerprint/.next_chunk_id logic "
                             "resume an interrupted generation in place. Only use this when you "
                             "deliberately want to discard whatever's already in that directory "
                             "(chunks, checkpoint, posterior, logs) and start over from nothing.")
    parser.add_argument('--total_n_sims', type=int, default=None,
                        help="If set, rename the dataset directory to SBI_prods_{total_n_sims} "
                             "after all stages complete (e.g. 350000 + 650000 → 1000000).")

    # Test simulation parameter overrides.
    #
    # Generic rather than one flag per parameter: the parameter set now comes
    # from the config, so a fixed list of --DM/--gamma_leading/... flags here
    # would silently ignore anything a new config declares (and offered
    # --tau_scatter and --sigma_bandpass, which simulation_maker.py no longer
    # takes as free parameters at all).
    #
    #   --set DM=0.01 --set snr=40
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                        dest='set_params',
                        help="Pin a test-simulation parameter, e.g. --set DM=0.01. "
                             "Repeatable. Names must match the config's inferred/"
                             "nuisance parameters (see simulation_maker.py --help).")
    parser.add_argument('--test_seed', type=int, default=None,
                        help="RNG seed for the test simulation, for reproducibility")

    args = parser.parse_args()

    # --------------------------------------------------
    # Resolve and validate the config once, up front, so a typo fails here
    # rather than 6 hours into a simulation run.
    # --------------------------------------------------
    from .pulsar_config import get_config, resolve_config_path

    config_path = os.path.abspath(resolve_config_path(args.config))
    cfg = get_config(config_path)
    cfg_flag = f" --config {config_path}"

    print(cfg.summary())

    for kv in args.set_params:
        if "=" not in kv:
            parser.error(f"--set expects NAME=VALUE, got '{kv}'")
        name = kv.split("=", 1)[0]
        declared = {p.name for p in cfg.inferred} | {p.name for p in cfg.nuisance}
        if name not in declared:
            parser.error(
                f"--set {kv}: '{name}' is not a parameter in "
                f"{os.path.basename(config_path)}. Declared: "
                f"{', '.join(sorted(declared))}"
            )

    # Expand --all
    if args.all:
        args.make_dataset = True
        args.train = True
        args.simul_valid = True
        args.make_test = True
        args.sample = True
        args.PPC = True
        args.pp_test = True

    if not any([args.make_dataset, args.train, args.simul_valid, args.val_loss_plot,
                args.make_test, args.sample, args.PPC, args.pp_test]):
        parser.error("No stage selected. Use --all or stage flags.")
    if args.PPC and not args.sample:
        parser.error("PPC stage requires --sample flag.")

    # --------------------------------------------------
    # Dataset directory (default SBI_prods_{n_sims})
    # --------------------------------------------------

    dataset_dir = args.dataset if args.dataset else f"SBI_prods_{args.n_sims}"
    print(f"Config: {config_path}  (fingerprint {cfg.fingerprint})")

    if args.make_dataset:
        if args.resume:
            # Archive existing contents into old_train/ so new chunks start fresh
            old_train_dir = os.path.join(dataset_dir, "old_train")
            if os.path.exists(old_train_dir):
                parser.error(f"{old_train_dir} already exists — looks like a resume was already done.")
            Path(dataset_dir).mkdir(exist_ok=True)
            os.makedirs(old_train_dir)
            for item in os.listdir(dataset_dir):
                if item != "old_train":
                    shutil.move(os.path.join(dataset_dir, item), old_train_dir)
            print(f"Resume mode: archived existing contents to {old_train_dir}")
        elif args.fresh_start:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            Path(dataset_dir).mkdir()
            print(f"--fresh_start: wiped {dataset_dir}")
        # else: leave dataset_dir exactly as it is. sim_maker.py's own
        # fingerprint + .next_chunk_id logic (see sim_maker.py's "Continue
        # chunk numbering" block) detects and resumes an interrupted
        # generation safely on its own -- this function must never clear
        # the directory out from under it by default. Doing so once already
        # destroyed 572 rounds of training and ~4.6M sims (2026-10-11),
        # because a concurrently-running --train --wait_for_chunks job had
        # no idea its checkpoint.pt and chunk files had just been deleted.

    # Ensures dataset_dir exists even when --make-dataset isn't part of this
    # invocation (e.g. a --train --wait_for_chunks job started before the
    # separate --make-dataset job has created it) — otherwise pipeline_log's
    # open() below fails before training gets anywhere near the dataset.
    # A no-op if --make-dataset already created it, and never touches existing
    # contents (unlike the rmtree above), so it's safe unconditionally.
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    pipeline_log = os.path.join(dataset_dir, "pipeline.log")

    print(f"\nUsing dataset directory: {dataset_dir}")
    print(f"Pipeline log: {pipeline_log}")

    # --------------------------------------------------
    # DATASET GENERATION
    # --------------------------------------------------
    if args.make_dataset:
        cmd = (f"{sys.executable} -u -m p2sbi.sim_maker{cfg_flag} --n_sims {args.n_sims}"
               f" --outdir {dataset_dir} --chunk_size {args.chunk_size}")
        if args.n_workers is not None:
            cmd += f" --n_workers {args.n_workers}"
        if args.max_outstanding_chunks is not None:
            cmd += f" --max_outstanding_chunks {args.max_outstanding_chunks}"
        run_step("DATASET GENERATION", cmd, dataset_dir, pipeline_log)

    # --------------------------------------------------
    # TRAINING
    # No --config here: train_sbi.py defaults to the config recorded in the
    # dataset's metadata.pt, which cannot drift from the data it was made from.
    # --------------------------------------------------
    if args.train:
        train_start_time = time.time()
        cmd = (f"{sys.executable} -u -m p2sbi.train_sbi --dataset {dataset_dir} --epochs {args.epochs}"
               f" --encoder {args.encoder}")
        if args.batch_size is not None:
            cmd += f" --batch_size {args.batch_size}"
        if args.wait_for_chunks:
            cmd += (f" --wait_for_chunks --poll_interval {args.poll_interval}"
                    f" --stall_timeout {args.stall_timeout}")
        if torch.cuda.is_available():
            sims_per_round = args.sims_per_round if args.sims_per_round is not None else 100000
            group_size = max(1, sims_per_round // args.chunk_size)
            cmd += f" --group_size {group_size}"
            print(f"CUDA detected: group_size={group_size} "
                  f"({group_size * args.chunk_size:,} sims/round, chunk_size={args.chunk_size:,})")
            if args.sims_per_round is None:
                print("WARNING: --sims_per_round not set, defaulting to the historical 100,000 "
                      "target — this may not fit your GPU. Verify with tools/probe_vram.py.")
        if args.seed is not None:
            cmd += f" --seed {args.seed}"
        if args.resume:
            resume_posterior = os.path.join(dataset_dir, "old_train", "posterior.pt")
            cmd += f" --resume {resume_posterior} --lr 1e-4"
            print(f"Resume mode: warm-starting from {resume_posterior}")
            print("Training will run only on the new chunks in this dataset directory.")
            print("Using lr=1e-4 (half of default) to reduce catastrophic forgetting.")
        if args.resume_checkpoint:
            cmd += f" --resume_checkpoint {args.resume_checkpoint}"
            print(f"Resuming interrupted run from checkpoint: {args.resume_checkpoint}")
        if args.delete_after_training:
            cmd += " --delete_after_training"
            print("Deletion mode: each round's chunks will be removed once checkpointed.")
        run_step("TRAINING", cmd, dataset_dir, pipeline_log)
        total_time = time.time() - train_start_time
        

    # --------------------------------------------------
    # VALIDATION LOSS PLOT
    # --------------------------------------------------
    if args.val_loss_plot:
        cmd = f"{sys.executable} -m p2sbi.val_loss_plot --dataset {dataset_dir}"
        run_step("VALIDATION LOSS PLOT", cmd, dataset_dir, pipeline_log)

    # --------------------------------------------------
    # SIMULATION VALIDATION (true-vs-recovered)
    # --------------------------------------------------
    if args.simul_valid:
        posterior = args.posterior or os.path.join(dataset_dir, "posterior.pt")
        cmd = (f"{sys.executable} -m p2sbi.simul_valid --dataset {dataset_dir}"
               f" --posterior {posterior}"
               f" --n_test {args.n_valid}"
               f" --n_samples {args.n_valid_samples}"
               f" --outdir {dataset_dir}")
        run_step("SIMULATION VALIDATION", cmd, dataset_dir, pipeline_log)

    # --------------------------------------------------
    # TEST SIMULATION
    # --------------------------------------------------
    if args.make_test:
        cmd = f"{sys.executable} -m p2sbi.simulation_maker{cfg_flag} --outdir {dataset_dir} --verbose"
        if args.test_seed is not None:
            cmd += f" --seed {args.test_seed}"
        if args.set_params:
            print("Pinned test sim parameters:")
            for kv in args.set_params:
                name, value = kv.split("=", 1)
                print(f"  {name}: {value}")
                cmd += f" --{name} {value}"
        else:
            print("No test sim parameters pinned — all drawn from the config's priors.")
        run_step("TEST SIMULATION", cmd, dataset_dir, pipeline_log)

    # --------------------------------------------------
    # SAMPLING
    # --------------------------------------------------
    if args.sample:

        posterior = args.posterior or os.path.join(dataset_dir, "posterior.pt")
        
        test_sim = args.test_sim or max(glob.glob(os.path.join(dataset_dir, "sim*.pt")),key=os.path.getmtime)


        print("Using posterior:", posterior)
        print("Using test sim:", test_sim)

        cmd = f"{sys.executable} -m p2sbi.DM_sampling --posterior_tensor {posterior} --test_simulation {test_sim} --outdir {dataset_dir}"
        run_step("POSTERIOR SAMPLING", cmd, dataset_dir, pipeline_log)

        # --------------------------------------------------
        # POSTERIOR PREDICTIVE CHECKS
        # --------------------------------------------------
        if args.PPC:
            print("\nPosterior sampling complete. Now running posterior predictive checks...")

            cmd = f"{sys.executable} -m p2sbi.PPC --posterior_tensor {posterior} --test_simulation {test_sim} --outdir {dataset_dir} --n_sims {args.n_sims}"
            run_step("POSTERIOR PREDICTIVE CHECKS", cmd, dataset_dir, pipeline_log)

    # --------------------------------------------------
    # PP TESTS
    # --------------------------------------------------
    if args.pp_test:
        posterior = args.posterior or os.path.join(dataset_dir, "posterior.pt")
        cmd = f"{sys.executable} -m p2sbi.pp_tests --posterior_tensor {posterior} --outdir {dataset_dir} --n_sims {args.n_sims} --n_test_cases {args.n_test_cases}"
        run_step("PP TESTS", cmd, dataset_dir, pipeline_log)

    # --------------------------------------------------
    # RENAME TO FINAL SIM COUNT
    # --------------------------------------------------
    if args.total_n_sims:
        final_dir = f"SBI_prods_{args.total_n_sims}"
        if os.path.abspath(dataset_dir) != os.path.abspath(final_dir):
            os.rename(dataset_dir, final_dir)
            print(f"\nRenamed {dataset_dir} → {final_dir}")
            dataset_dir = final_dir

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()