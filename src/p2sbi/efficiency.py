from dataclasses import dataclass, field
from typing import List
import time
import json
import torch
import platform
import psutil

@dataclass
class ChunkStats:
    chunk_id: int
    num_simulations: int
    epochs_trained: int
    best_val_loss: float
    time_seconds: float
    cumulative_simulations: int

@dataclass 
class RoundStats:
    round_id: int
    chunks: List[ChunkStats] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_chunk(self, chunk: ChunkStats):
        self.chunks.append(chunk)

    @property
    def total_time(self):
        return sum(c.time_seconds for c in self.chunks)

    @property
    def total_epochs(self):
        return sum(c.epochs_trained for c in self.chunks)

    @property
    def total_simulations(self):
        return sum(c.num_simulations for c in self.chunks)

    @property
    def final_val_loss(self):
        # What matters is where you ended up, not the average
        return self.chunks[-1].best_val_loss if self.chunks else None

    @property
    def loss_improvement(self):
        # How much did the loss improve from first to last chunk
        if len(self.chunks) < 2:
            return 0.0
        return self.chunks[-1].best_val_loss - self.chunks[0].best_val_loss
    


def compute_round_efficiency(round_stats: RoundStats, hardware_profile: dict):
    
    # Simulations processed per second across entire round
    sims_per_second = round_stats.total_simulations / round_stats.total_time

    # How much loss improved per unit time — this is the key metric
    # More negative final loss = better, improvement should be negative (going down)
    loss_improvement_per_second = round_stats.loss_improvement / round_stats.total_time

    # Epoch efficiency: are later chunks converging faster (healthy transfer)?
    epoch_counts = [c.epochs_trained for c in round_stats.chunks]
    transfer_efficiency = epoch_counts[0] / epoch_counts[-1] if epoch_counts[-1] > 0 else 1.0
    # > 1 means later chunks needed fewer epochs → good weight transfer

    # Average convergence rate per chunk
    per_chunk_convergence = [
        abs(c.best_val_loss) / c.time_seconds 
        for c in round_stats.chunks
    ]

    score = {
        "round_id": round_stats.round_id,
        "hardware": hardware_profile,
        "cumulative": {
            "total_simulations": round_stats.total_simulations,
            "total_epochs": round_stats.total_epochs,
            "total_time_seconds": round(round_stats.total_time, 2),
            "final_val_loss": round_stats.final_val_loss,
            "loss_improvement": round(round_stats.loss_improvement, 4),
        },
        "efficiency": {
            "sims_per_second": round(sims_per_second, 4),
            "loss_improvement_per_second": round(loss_improvement_per_second, 6),
            "transfer_efficiency": round(transfer_efficiency, 4),
            "per_chunk_convergence": [round(x, 4) for x in per_chunk_convergence],
        }
    }
    return score


def get_hardware_profile():
    profile = {
        "platform": platform.system(),
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "device": None,
        "device_name": None,
    }

    if torch.cuda.is_available():
        profile["device"] = "cuda"
        profile["device_name"] = torch.cuda.get_device_name(0)
        profile["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    elif torch.backends.mps.is_available():
        profile["device"] = "mps"
        profile["device_name"] = "Apple MPS"
        # MPS shares RAM, so report system RAM
        profile["vram_gb"] = "shared"
    else:
        profile["device"] = "cpu"
        profile["device_name"] = "CPU only"

    return profile