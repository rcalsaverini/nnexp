import pytest
import numpy as np
import sympy as sp
import tempfile
from pathlib import Path

from nnexp.grammar.ebnf import MATH_TOKENS, MathGrammarSpec, TokenType
from nnexp.data.generator import ASTGenerator, PointCloudSampler
from nnexp.data.dataset import save_dataset, load_dataset


def test_ebnf_token_types():
    assert MathGrammarSpec.get_token_type("x") == TokenType.OPERAND
    assert MathGrammarSpec.get_token_type("+") == TokenType.BINARY_OPERATOR
    assert MathGrammarSpec.get_token_type("sin") == TokenType.UNARY_FUNCTION
    assert MathGrammarSpec.get_token_type("(") == TokenType.OPEN_PAREN
    assert MathGrammarSpec.get_token_type(")") == TokenType.CLOSE_PAREN


def test_ast_generation():
    generator = ASTGenerator(seed=123)
    expr = generator.generate_expression(max_depth=3)
    assert isinstance(expr, sp.Expr)
    assert str(expr) != ""


def test_point_cloud_sampling():
    x = sp.Symbol("x")
    expr = sp.sin(x) + 2 * x
    sampler = PointCloudSampler(num_points=50)
    result = sampler.sample_points(expr)

    assert result is not None
    x_grid, y_grid = result
    assert len(x_grid) == 50
    assert len(y_grid) == 50
    assert np.all(np.isfinite(y_grid))


def test_dataset_saving_loading():
    samples = [
        {"expression": "sin(x)", "x": np.linspace(-1, 1, 10), "y": np.sin(np.linspace(-1, 1, 10))},
        {"expression": "x**2", "x": np.linspace(-1, 1, 10), "y": np.linspace(-1, 1, 10)**2},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"
        save_dataset(samples, parquet_path)
        assert parquet_path.exists()

        df = load_dataset(parquet_path)
        assert len(df) == 2
        assert "expression" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
