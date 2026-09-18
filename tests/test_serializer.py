import numpy as np
import pytest
import sympy as sp

from nnexp.data.generator import ASTGenerator, PointCloudSampler
from nnexp.grammar.ebnf import EOS_TOKEN, MATH_TOKENS
from nnexp.grammar.serializer import (
    UnsupportedExpressionError,
    expression_from_tokens,
    tokenize_expression,
    tokens_match_points,
    tokens_to_string,
)

x = sp.Symbol("x")


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x, "x"),
        (sp.Integer(15), "15"),
        (sp.Integer(-5), "-5"),
        (sp.Float(0.5), "0.5"),
        (sp.Rational(1, 2), "1/2"),
        (sp.pi, "pi"),
        (sp.E, "e"),
        (x + 1, "x+1"),
        (x - 1, "x-1"),
        (-x, "-x"),
        (2 * x, "2*x"),
        (x / 2, "x/2"),
        (-x / 2, "-x/2"),
        (x**2, "x^2"),
        (sp.sqrt(x), "sqrt(x)"),
        (sp.Abs(x), "abs(x)"),
        (sp.sin(x), "sin(x)"),
        (sp.exp(x), "exp(x)"),
        ((x + 1) ** 2, "(x+1)^2"),
        ((-4) ** x, "(-4)^x"),
        (1 / (x + 1), "1/(x+1)"),
        (x / (2 * x), "1/2"),
    ],
)
def test_renders_expected_token_string(expr, expected):
    assert tokens_to_string(tokenize_expression(expr)) == expected


def test_every_token_is_in_the_math_vocabulary():
    expr = sp.sin(x) / (sp.sqrt(sp.Abs(x)) + 2.5) - 15 * x**3
    assert set(tokenize_expression(expr)) <= set(MATH_TOKENS)


def test_sympy_str_is_not_tokenizable_but_our_rendering_is():
    """The regression this module exists for: '**', 'E' and 'Abs' are not tokens."""
    expr = sp.Abs(x) ** sp.E
    assert not set(str(expr)) <= set(MATH_TOKENS)
    assert set(tokenize_expression(expr)) <= set(MATH_TOKENS)


def test_include_eos_appends_terminator():
    tokens = tokenize_expression(x + 1, include_eos=True)
    assert tokens[-1] == EOS_TOKEN
    assert tokens_to_string(tokens) == "x+1"


@pytest.mark.parametrize(
    "expr",
    [
        sp.sqrt(2) * sp.I,
        x + sp.I,
        sp.Symbol("y") + x,
        sp.re(x),
        sp.zoo * x,
    ],
)
def test_rejects_expressions_outside_the_vocabulary(expr):
    with pytest.raises(UnsupportedExpressionError):
        tokenize_expression(expr)


def test_symbols_carrying_assumptions_are_accepted():
    x_real = sp.Symbol("x", real=True)
    assert tokens_to_string(tokenize_expression(x_real + 1)) == "x+1"


@pytest.mark.parametrize(
    "expr",
    [
        x**2 - 3 * x + 1,
        sp.sin(x) / (x**2 + 1),
        sp.exp(-(x**2)),
        (x + 1) ** 3 / (x - 5),
        sp.Abs(x) ** sp.Rational(3, 2),
        -sp.log(sp.sqrt(x**2 + 1)),
        15 * x + 0.25,
        sp.pi * sp.cos(2 * x) - sp.E,
    ],
)
def test_roundtrip_preserves_the_function(expr):
    tokens = tokenize_expression(expr, include_eos=True)
    reparsed = expression_from_tokens(tokens)
    assert sp.simplify(reparsed - expr) == 0


def test_tokens_match_points_accepts_a_faithful_rendering():
    expr = sp.sin(x) + 2 * x
    sampler = PointCloudSampler(num_points=50)
    x_grid, y_grid = sampler.sample_points(expr)
    assert tokens_match_points(tokenize_expression(expr, include_eos=True), x_grid, y_grid)


def test_tokens_match_points_rejects_a_different_function():
    sampler = PointCloudSampler(num_points=50)
    x_grid, y_grid = sampler.sample_points(sp.sin(x) + 2 * x)
    assert not tokens_match_points(tokenize_expression(sp.cos(x)), x_grid, y_grid)


def test_generated_expressions_roundtrip_through_the_vocabulary():
    """End-to-end: whatever the generator emits either renders faithfully or is rejected."""
    generator = ASTGenerator(seed=7)
    sampler = PointCloudSampler(num_points=50)

    verified = 0
    attempts = 0
    while verified < 150 and attempts < 5000:
        attempts += 1
        expr = generator.generate_expression(max_depth=4)
        points = sampler.sample_points(expr)
        if points is None:
            continue
        try:
            tokens = tokenize_expression(expr, include_eos=True)
        except UnsupportedExpressionError:
            continue
        x_grid, y_grid = points
        assert set(tokens) <= set(MATH_TOKENS)
        # A rendering that parses but denotes a different function is a defect.
        assert tokens_match_points(tokens, x_grid, y_grid), (
            f"rendering diverged: {expr} -> {tokens_to_string(tokens)}"
        )
        verified += 1

    assert verified == 150, f"only verified {verified} expressions in {attempts} attempts"
