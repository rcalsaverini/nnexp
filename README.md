# Neural Networks for Symbolic Optimization (SymReg)

This repository serves as a research sandbox for exploring the intersection of discrete symbolic manipulation and continuous deep learning optimization. The primary objective is to map combinatorial optimization problems into continuous vector spaces to leverage the representational power and gradient-based training of modern neural architectures.

## The Core Problem
Many valuable tasks—such as finding optimal regular expressions, symbolic regression, or generating syntactic code—require searching through a combinatorially large, discrete space of Abstract Syntax Trees (ASTs). Pure enumeration is computationally unfeasible, and traditional continuous optimization techniques (like standard gradient descent) fail when interacting with discrete sampling bottlenecks.

## Infrastructure & Workflow
This repository is optimized for a low-friction, script-only workflow. It is intended to be used to run external training at RunPod for experiments with minimal manual work.

### Technology Stack
- `uv` for dependency management & virtualenv locking
- `skypilot` for cloud orchestration (on `runpod`)
- `pytorch` for deep learning
- `wandb` for experiment tracking & telemetry
- `transformers` & `peft` (QLoRA) for LLM fine-tuning
- `trl` (`PPOTrainer`) for reinforcement learning

---

### Local Environment Setup

```bash
# 1. Install/sync virtual environment
uv sync

# 2. Run local sanity check (hardware detection, tensor ops, offline WandB)
uv run python scripts/sanity_check.py --offline
```

---

### Cloud Authentication Setup (One-Time)

```bash
# 1. Authenticate RunPod API Key (saves to ~/.runpod/config.toml)
uv run runpod config

# 2. Authenticate Weights & Biases (saves to ~/.netrc)
uv run wandb login

# 3. Verify cloud provider readiness
uv run sky check
```

---

### SkyPilot Operations & Cluster Lifecycle

#### 1. Launching Tasks
```bash
# Export WandB API key from ~/.netrc
export WANDB_API_KEY=$(grep "password" ~/.netrc | awk '{print $2}')

# Launch task on RunPod (provisions GPU instance automatically)
uv run sky launch --env WANDB_API_KEY=$WANDB_API_KEY task.yaml
```

#### 2. Autostop & Auto-Teardown
```bash
# Auto-teardown (delete cluster immediately after task completes)
uv run sky launch --down --env WANDB_API_KEY=$WANDB_API_KEY task.yaml

# Auto-stop cluster after 10 minutes of idle time
uv run sky launch --idle-minutes-to-autostop 10 --env WANDB_API_KEY=$WANDB_API_KEY task.yaml
```

#### 3. Reusing & Autostarting Existing Clusters
```bash
# Launch/execute on an existing cluster (autostarts if stopped via `sky stop`)
uv run sky launch --cluster <cluster_name> --env WANDB_API_KEY=$WANDB_API_KEY task.yaml

# Execute a new task on a running cluster without re-provisioning
uv run sky exec <cluster_name> --env WANDB_API_KEY=$WANDB_API_KEY task.yaml
```

#### 4. Cluster Management Commands
```bash
# View active clusters
uv run sky status -a

# View job status and logs
uv run sky queue
uv run sky logs <cluster_name> <job_id>

# Teardown / delete a cluster completely (RunPod uses `sky down` for termination)
uv run sky down <cluster_name>

# Teardown all clusters
uv run sky down -a
```

---

## Documentation
For a deep dive into the theoretical framework and project architecture, see `docs/PROGRAM.md`.
For the specifics of the current active trial, see `docs/EXPERIMENTS/symbolic_regression.md`.