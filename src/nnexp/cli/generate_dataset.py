import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from nnexp.data import ASTGenerator, PointCloudSampler, save_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic mathematical expression dataset")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of valid samples to generate")
    parser.add_argument("--num-points", type=int, default=50, help="Number of points in point cloud (N)")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum AST expression depth")
    parser.add_argument("--out", type=str, default="data/synthetic_dataset.parquet", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== Generating Synthetic Dataset ({args.num_samples} samples) ===")
    generator = ASTGenerator(seed=args.seed)
    sampler = PointCloudSampler(num_points=args.num_points)

    samples = []
    attempts = 0
    max_attempts = args.num_samples * 20

    pbar = tqdm(total=args.num_samples, desc="Generating valid ASTs")
    while len(samples) < args.num_samples and attempts < max_attempts:
        attempts += 1
        expr = generator.generate_expression(max_depth=args.max_depth)
        result = sampler.sample_points(expr)

        if result is not None:
            x_pts, y_pts = result
            samples.append({
                "expression": str(expr),
                "x": x_pts,
                "y": y_pts,
            })
            pbar.update(1)

    pbar.close()

    if len(samples) < args.num_samples:
        print(f"Warning: Only generated {len(samples)} valid samples out of {attempts} attempts.")

    out_path = save_dataset(samples, args.out)
    print(f"Dataset successfully saved to: {out_path.resolve()}")
    print(f"Total samples saved: {len(samples)}")


if __name__ == "__main__":
    main()
