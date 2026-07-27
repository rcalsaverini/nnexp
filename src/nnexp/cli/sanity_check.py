import argparse
import sys
import torch
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 Sanity Check")
    parser.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Phase 0 Telemetry & PyTorch Sanity Check ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    x = torch.randn(10, 2, device=device)
    w = torch.randn(2, 64, device=device)
    y = torch.matmul(x, w)
    print(f"Tensor computation check: Input shape {x.shape} -> Output shape {y.shape}")

    mode = "offline" if args.offline else "online"
    print(f"Initializing WandB run (mode={mode})...")
    run = wandb.init(
        project="nnexp-symreg",
        name="phase0-sanity-check",
        mode=mode,
        config={
            "pytorch_version": torch.__version__,
            "device": str(device),
        },
    )
    wandb.log({"sanity_check_passed": 1, "test_loss": 0.042})
    wandb.finish()
    print("WandB run completed successfully.")
    print("=== Sanity Check Passed ===")


if __name__ == "__main__":
    main()
