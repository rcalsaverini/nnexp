"""
Serialization between SymPy expressions and the project's math token alphabet.

SymPy's own string form is not expressible in `MATH_TOKENS`: it emits `**` instead
of `^`, `E` instead of `e`, `Abs` instead of `abs`, and can emit symbols with no
token at all (`I`, `zoo`). This module renders an expression directly from its AST
into the grammar alphabet, and parses it back for round-trip verification.
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .ebnf import EOS_TOKEN, MATH_TOKENS

__all__ = [
    "UnsupportedExpressionError",
    "tokenize_expression",
    "tokens_to_string",
    "expression_from_tokens",
    "tokens_match_points",
]


class UnsupportedExpressionError(ValueError):
    """Raised when an expression cannot be rendered in the math vocabulary."""


# Operator precedence, used to decide where parentheses are required.
PREC_ADD = 1
PREC_MUL = 2
PREC_POW = 3
PREC_ATOM = 4

# Guards against pathological recursion in `_emit`.
MAX_DEPTH = 64

_VOCAB = set(MATH_TOKENS)

# SymPy class name -> grammar function token. `sqrt` is absent because SymPy
# represents it as Pow(base, 1/2); it is special-cased in `_emit`.
_FUNCTION_TOKENS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "Abs": "abs",
}

_X = sp.Symbol("x")

# Closed name table for parsing token strings back: every name the emitter can
# produce, and nothing else.
_PARSE_LOCALS = {
    "x": _X,
    "e": sp.E,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


def tokenize_expression(expr: sp.Expr, include_eos: bool = False) -> List[str]:
    """
    Renders a SymPy expression as a list of tokens drawn from `MATH_TOKENS`.

    Args:
        expr: Single-variable SymPy expression over the symbol `x`.
        include_eos: Append the `<EOS>` token, for use as a generation target.

    Returns:
        Token list, every element a member of `MATH_TOKENS`.

    Raises:
        UnsupportedExpressionError: The expression contains nodes, symbols or
            numeric literals with no representation in the math vocabulary.
    """
    expr = sp.sympify(expr)

    if expr.has(sp.I, sp.zoo, sp.oo, sp.nan):
        raise UnsupportedExpressionError(f"non-real or non-finite expression: {expr}")

    foreign = sorted({s.name for s in expr.free_symbols if s.name != "x"})
    if foreign:
        raise UnsupportedExpressionError(f"unsupported free symbols: {foreign}")

    tokens, _ = _emit(expr, 0)

    unknown = sorted({t for t in tokens if t not in _VOCAB})
    if unknown:
        raise UnsupportedExpressionError(f"tokens outside math vocabulary: {unknown}")

    if include_eos:
        tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_string(tokens: Iterable[str]) -> str:
    """Joins tokens into an expression string, dropping `<EOS>`."""
    return "".join(t for t in tokens if t != EOS_TOKEN)


def expression_from_tokens(tokens: Iterable[str]) -> sp.Expr:
    """
    Parses a token sequence back into a SymPy expression.

    Names resolve against `_PARSE_LOCALS` only, which covers exactly the symbols
    and functions the emitter can produce.
    """
    text = tokens_to_string(tokens).replace("^", "**")
    if not text:
        raise UnsupportedExpressionError("empty token sequence")
    return parse_expr(text, local_dict=_PARSE_LOCALS, evaluate=True)


def tokens_match_points(
    tokens: Sequence[str],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    """
    Round-trip check: re-parses `tokens` and confirms the resulting expression
    reproduces the sampled point cloud. Guards against a token rendering that
    parses cleanly but denotes a different function.
    """
    x_grid = np.asarray(x_grid, dtype=np.float64)
    try:
        expr = expression_from_tokens(tokens)
        fn = sp.lambdify(_X, expr, modules=["numpy"])
        y_hat = np.asarray(fn(x_grid))

        if np.iscomplexobj(y_hat):
            return False
        if y_hat.ndim == 0:
            y_hat = np.full_like(x_grid, y_hat.item())
        y_hat = y_hat.astype(np.float64)
    except Exception:
        return False

    if y_hat.shape != np.shape(y_grid):
        return False
    return bool(np.allclose(y_hat, y_grid, rtol=rtol, atol=atol, equal_nan=False))


def _emit_wrapped(expr: sp.Expr, min_prec: int, depth: int) -> List[str]:
    """Emits `expr`, parenthesized if its precedence is below `min_prec`."""
    tokens, prec = _emit(expr, depth)
    if prec < min_prec:
        return ["("] + tokens + [")"]
    return tokens


def _emit(expr: sp.Expr, depth: int) -> Tuple[List[str], int]:
    """Recursively renders `expr`, returning its tokens and its precedence."""
    if depth > MAX_DEPTH:
        raise UnsupportedExpressionError(f"expression nesting exceeds {MAX_DEPTH} levels")
    d = depth + 1

    # --- Atoms ---
    if expr == sp.pi:
        return ["pi"], PREC_ATOM
    if expr == sp.E:
        return ["e"], PREC_ATOM
    if isinstance(expr, sp.Symbol):
        # Matched by name so that symbols carrying assumptions (real=True, ...)
        # are accepted too; SymPy treats those as distinct objects.
        if expr.name != "x":
            raise UnsupportedExpressionError(f"unsupported symbol: {expr.name!r}")
        return ["x"], PREC_ATOM
    if isinstance(expr, sp.Integer):
        value = int(expr)
        if value >= 0:
            return list(str(value)), PREC_ATOM
    if isinstance(expr, sp.Float):
        value = float(expr)
        if value >= 0:
            return _float_tokens(value), PREC_ATOM

    # --- Leading unary minus: -(...) ---
    # Runs before the fraction split so that -x/2 renders as -x/2, not (-x)/2.
    if expr.could_extract_minus_sign():
        return ["-"] + _emit_wrapped(-expr, PREC_MUL, d), PREC_ADD

    # --- Division: a/b. Also covers Rational and negative powers. ---
    numerator, denominator = sp.fraction(expr)
    if denominator != sp.S.One:
        left = _emit_wrapped(numerator, PREC_MUL, d)
        right = _emit_wrapped(denominator, PREC_POW, d)
        return left + ["/"] + right, PREC_MUL

    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if exponent == sp.Rational(1, 2):
            return ["sqrt", "("] + _emit_wrapped(base, PREC_ADD, d) + [")"], PREC_ATOM
        # Both operands sit at atom level: the grammar's `factor` rule takes a
        # `primary` on each side of `^`, and `a^b^c` would otherwise be ambiguous.
        base_tokens = _emit_wrapped(base, PREC_ATOM, d)
        exponent_tokens = _emit_wrapped(exponent, PREC_ATOM, d)
        return base_tokens + ["^"] + exponent_tokens, PREC_POW

    if isinstance(expr, sp.Add):
        terms = expr.as_ordered_terms()
        tokens = _emit_wrapped(terms[0], PREC_ADD, d)
        for term in terms[1:]:
            if term.could_extract_minus_sign():
                tokens += ["-"] + _emit_wrapped(-term, PREC_MUL, d)
            else:
                tokens += ["+"] + _emit_wrapped(term, PREC_MUL, d)
        return tokens, PREC_ADD

    if isinstance(expr, sp.Mul):
        factors = expr.as_ordered_factors()
        tokens = _emit_wrapped(factors[0], PREC_MUL, d)
        for factor in factors[1:]:
            tokens += ["*"] + _emit_wrapped(factor, PREC_MUL, d)
        return tokens, PREC_MUL

    name = _FUNCTION_TOKENS.get(type(expr).__name__)
    if name is not None and len(expr.args) == 1:
        return [name, "("] + _emit_wrapped(expr.args[0], PREC_ADD, d) + [")"], PREC_ATOM

    raise UnsupportedExpressionError(f"unsupported expression node: {sp.srepr(expr)[:120]}")


def _float_tokens(value: float) -> List[str]:
    """Renders a non-negative float as digit and '.' tokens."""
    text = repr(value)
    if not all(c in "0123456789." for c in text):
        # Scientific notation, inf, nan: no representation in the grammar.
        raise UnsupportedExpressionError(f"float literal not representable: {text}")
    if "." not in text:
        text += ".0"
    return list(text)
