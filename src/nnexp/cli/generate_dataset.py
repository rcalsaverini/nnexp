import argparse
from collections import Counter

from tqdm import tqdm

from nnexp.data import ASTGenerator, PointCloudSampler, save_dataset
from nnexp.grammar.serializer import (
    UnsupportedExpressionError,
    tokenize_expression,
    tokens_match_points,
    tokens_to_string,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic mathematical expression dataset")
    parser.add_argument("--num-samples", type=int, default=100_000, help="Number of valid samples to generate")
    parser.add_argument("--num-points", type=int, default=50, help="Number of points in point cloud (N)")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum AST expression depth")
    parser.add_argument("--out", type=str, default="data/synthetic_dataset.parquet", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the token round-trip check (faster, but tokens are no longer guaranteed "
        "to reproduce the stored point cloud)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    verify = not args.no_verify

    print(f"=== Generating Synthetic Dataset ({args.num_samples} samples) ===")
    generator = ASTGenerator(seed=args.seed)
    sampler = PointCloudSampler(num_points=args.num_points)

    samples = []
    rejected = Counter()
    attempts = 0
    max_attempts = args.num_samples * 20

    pbar = tqdm(total=args.num_samples, desc="Generating valid ASTs")
    while len(samples) < args.num_samples and attempts < max_attempts:
        attempts += 1

        # SymPy can exhaust the Python stack inside its own arithmetic while
        # folding pathological constants. Counted rather than swallowed, so a
        # regression here stays visible instead of quietly shrinking the run.
        try:
            expr = generator.generate_expression(max_depth=args.max_depth)
            points = sampler.sample_points(expr)
        except RecursionError:
            rejected["RecursionError during construction"] += 1
            continue

        if points is None:
            rejected["not sampleable (complex/non-finite/constant)"] += 1
            continue
        x_pts, y_pts = points

        # Render into the math vocabulary the grammar mask operates over. SymPy's
        # own string form uses '**', 'E' and 'Abs', none of which are tokens.
        try:
            tokens = tokenize_expression(expr, include_eos=True)
        except UnsupportedExpressionError:
            rejected["not representable in math vocabulary"] += 1
            continue

        # Confirm the rendered tokens parse back to the same function.
        if verify and not tokens_match_points(tokens, x_pts, y_pts):
            rejected["token round-trip mismatch"] += 1
            continue

        samples.append({
            "expression": str(expr),
            "tokens": tokens,
            "token_string": tokens_to_string(tokens),
            "x": x_pts,
            "y": y_pts,
        })
        pbar.update(1)

    pbar.close()

    if len(samples) < args.num_samples:
        print(f"Warning: Only generated {len(samples)} valid samples out of {attempts} attempts.")

    out_path = save_dataset(samples, args.out)

    print(f"\nDataset successfully saved to: {out_path.resolve()}")
    print(f"Total samples saved: {len(samples)} (from {attempts} attempts)")
    print(f"Round-trip verification: {'on' if verify else 'OFF (--no-verify)'}")
    if rejected:
        print("Rejected candidates:")
        for reason, count in rejected.most_common():
            print(f"  {count:>8}  {reason}")


if __name__ == "__main__":
    main()
