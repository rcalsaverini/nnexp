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


def test_power_over_negative_base_stays_real():
    """`cos((-2)**pi)**(x + 2)` used to raise RecursionError inside SymPy."""
    generator = ASTGenerator(seed=0)
    x = sp.Symbol("x")

    result = generator.apply_power(sp.cos(sp.Integer(-2) ** sp.pi), x + 2)
    assert not result.has(sp.I)

    for base in [sp.Integer(-2), sp.Integer(-5), sp.cos(sp.Integer(3))]:
        for exponent in [sp.pi, sp.E, sp.Rational(1, 2), x, x + 2]:
            assert not generator.apply_power(base, exponent).has(sp.I)


def test_power_keeps_integer_exponents_and_positive_bases():
    generator = ASTGenerator(seed=0)
    x = sp.Symbol("x")
    assert generator.apply_power(x, sp.Integer(2)) == x**2
    assert generator.apply_power(sp.Integer(-2), sp.Integer(3)) == sp.Integer(-8)
    assert generator.apply_power(sp.pi, x) == sp.pi**x


def test_log_and_sqrt_of_negative_numbers_stay_real():
    generator = ASTGenerator(seed=0)
    assert generator.apply_unary(sp.sqrt, sp.Integer(-5)) == sp.sqrt(5)
    assert generator.apply_unary(sp.log, sp.Integer(-5)) == sp.log(5)
    # A symbolic argument is a legal expression and must be left intact.
    x = sp.Symbol("x")
    assert generator.apply_unary(sp.sqrt, x) == sp.sqrt(x)


def test_sqrt_of_a_negative_transcendental_constant_stays_real():
    """`sqrt(tan(E))` is complex but carries no `I` node, so SymPy's assumption
    system cannot rule it out; wrapping it in Abs later then hangs."""
    generator = ASTGenerator(seed=0)
    result = generator.apply_unary(sp.sqrt, sp.tan(sp.E))
    assert complex(result.evalf(15)).imag == 0.0
    assert result == sp.sqrt(-sp.tan(sp.E))


def test_abs_over_a_reflected_term_terminates():
    """Regression: generation used to hang building Abs(-1 + E - sqrt(tan(E)))."""
    generator = ASTGenerator(seed=0)
    inner = (sp.Integer(-1) + sp.E) - generator.apply_unary(sp.sqrt, sp.tan(sp.E))
    result = generator.apply_unary(sp.Abs, inner)
    assert complex(result.evalf(15)).imag == 0.0


def test_generation_is_free_of_imaginary_units_and_crashes():
    generator = ASTGenerator(seed=2024)
    for _ in range(20_000):
        expr = generator.generate_expression(max_depth=4)
        assert not expr.has(sp.I)


def test_sampler_default_span_is_symmetric():
    sampler = PointCloudSampler(num_points=10)
    assert sampler.x_min == -5.0
    assert sampler.x_max == 5.0


def test_sampler_rejects_an_empty_span():
    with pytest.raises(ValueError):
        PointCloudSampler(x_min=5.0, x_max=-5.0)


def test_complex_valued_expressions_are_rejected_not_truncated():
    """A complex result must be dropped, never cast to float with the imaginary part lost."""
    x = sp.Symbol("x")
    expr = 2 * x + sp.sqrt(2) * sp.I
    assert PointCloudSampler(num_points=50).sample_points(expr) is None


def test_sampled_targets_are_real_and_finite():
    generator = ASTGenerator(seed=99)
    sampler = PointCloudSampler(num_points=50)

    checked = 0
    for _ in range(2000):
        result = sampler.sample_points(generator.generate_expression(max_depth=4))
        if result is None:
            continue
        _, y_grid = result
        assert y_grid.dtype == np.float64
        assert not np.iscomplexobj(y_grid)
        assert np.all(np.isfinite(y_grid))
        checked += 1
        if checked == 200:
            break
    assert checked == 200


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
