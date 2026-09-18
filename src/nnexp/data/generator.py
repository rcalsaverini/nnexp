import random
from typing import Tuple, Optional, List
import numpy as np
import sympy as sp


# Numeric exponents are kept small so powers stay cheap to evaluate and do not
# fold multi-thousand-digit integers into the expression.
MAX_ABS_EXPONENT = 5


class ASTGenerator:
    """Procedural AST generator for single-variable mathematical expressions."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.x = sp.Symbol("x")
        self.binary_ops = [
            sp.Add,
            sp.Mul,
            lambda a, b: a - b,
            lambda a, b: a / b if b != 0 else a,
            self.apply_power,
        ]
        self.unary_funcs = [
            sp.sin,
            sp.cos,
            sp.tan,
            sp.exp,
            sp.log,
            sp.sqrt,
            sp.Abs,
        ]

    def apply_power(self, base: sp.Expr, exponent: sp.Expr) -> sp.Expr:
        """
        Raises `base` to `exponent`, falling back to multiplication when the
        power would not be real.

        SymPy evaluates a negative base under a non-integer exponent into a
        complex number at construction time, and nesting those exhausts the
        Python stack while building the tree (e.g. `cos((-2)**pi)**(x + 2)`
        raises RecursionError inside SymPy's own arithmetic).
        """
        candidate = base**exponent if self._power_is_real(base, exponent) else base * exponent
        return self._reject_imaginary(candidate, base)

    def apply_unary(self, func, arg: sp.Expr) -> sp.Expr:
        """
        Applies a unary function, keeping `sqrt` and `log` over a non-negative
        argument. SymPy folds the imaginary unit straight into the expression
        for those (`sqrt(-5)` becomes `sqrt(5)*I`), and it has no grammar token.

        A symbolic argument is left intact: `sqrt(x)` is a legal expression, and
        the point sampler drops the samples where it turns non-finite.
        """
        if func in (sp.log, sp.sqrt):
            arg = self._reflect_negative_number(arg)
        result = func(arg)
        if not result.has(sp.I):
            return result
        return self._reject_imaginary(func(sp.Abs(arg)), arg)

    @staticmethod
    def _reflect_negative_number(arg: sp.Expr) -> sp.Expr:
        """
        Negates a numeric argument that evaluates below zero.

        The sign has to be decided numerically rather than through SymPy's
        assumption system, which leaves a transcendental constant such as
        `tan(E)` undetermined. That makes `sqrt(tan(E))` complex while carrying
        no explicit `I` node, and folding such a term into `Abs(...)` later
        sends SymPy into an exact-reasoning path that does not terminate in any
        practical time.
        """
        if not arg.is_number:
            return arg
        try:
            value = complex(arg.evalf(15))
        except (TypeError, ValueError):
            return arg
        if value.imag != 0.0:
            # Already complex: leave it for the sampler and tokenizer to reject
            # rather than building an Abs node over it.
            return arg
        return -arg if value.real < 0 else arg

    @staticmethod
    def _power_is_real(base: sp.Expr, exponent: sp.Expr) -> bool:
        if exponent.is_Integer:
            # Cap the magnitude to keep powers cheap: 5**3125 is a 2185-digit int.
            return abs(int(exponent)) <= MAX_ABS_EXPONENT
        if exponent.is_number and not exponent.is_real:
            return False
        # Non-integer exponent: real only over a provably non-negative base.
        return bool(base.is_nonnegative)

    @staticmethod
    def _reject_imaginary(candidate: sp.Expr, fallback: sp.Expr) -> sp.Expr:
        """
        Last-resort guard. SymPy's assumption system cannot always decide the
        sign of a transcendental constant, so `is_nonnegative` returning None is
        not proof the node is real: check the built expression itself.
        """
        return candidate if not candidate.has(sp.I) else fallback

    def generate_terminal(self) -> sp.Expr:
        choice = self.rng.choice(["x", "constant", "integer"])
        if choice == "x":
            return self.x
        elif choice == "constant":
            return self.rng.choice([sp.pi, sp.E])
        else:
            # Small integer coeff (-5 to 5, non-zero)
            val = self.rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            return sp.Integer(val)

    def generate_expression(self, max_depth: int = 3, depth: int = 0) -> sp.Expr:
        if depth >= max_depth:
            return self.generate_terminal()

        # Decide node type: terminal (30%), unary (30%), binary (40%)
        roll = self.rng.random()
        if roll < 0.3:
            return self.generate_terminal()
        elif roll < 0.6:
            func = self.rng.choice(self.unary_funcs)
            arg = self.generate_expression(max_depth, depth + 1)
            return self.apply_unary(func, arg)
        else:
            op = self.rng.choice(self.binary_ops)
            left = self.generate_expression(max_depth, depth + 1)
            right = self.generate_expression(max_depth, depth + 1)
            return op(left, right)


class PointCloudSampler:
    """Evaluates SymPy expressions over a grid to generate point cloud matrices X in R^(2 x N)."""

    def __init__(
        self,
        num_points: int = 50,
        x_min: float = -5.0,
        x_max: float = 5.0,
        variance_threshold: float = 1e-5,
    ):
        if x_max <= x_min:
            raise ValueError(f"x_max ({x_max}) must be greater than x_min ({x_min})")
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.variance_threshold = variance_threshold
        self.x_symbol = sp.Symbol("x")

    def sample_points(self, expr: sp.Expr) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples N points for expression y = f(x).
        Returns tuple of (x_grid, y_grid) if evaluation is valid & finite, else None.
        """
        # Ensure expression depends on x
        if not expr.has(self.x_symbol):
            return None

        # Negative bases raised to fractional powers make SymPy fold the imaginary
        # unit into the expression itself; reject before the cost of lambdify.
        if expr.has(sp.I):
            return None

        x_grid = np.linspace(self.x_min, self.x_max, self.num_points)
        try:
            # Lambda conversion to numpy
            fn = sp.lambdify(self.x_symbol, expr, modules=["numpy"])
            y_grid = np.asarray(fn(x_grid))

            # Reject complex results *before* any float cast. Casting complex to
            # float64 discards the imaginary part, so a check placed afterwards
            # can never fire and silently admits wrong targets.
            if np.iscomplexobj(y_grid):
                return None

            # Handle scalar output from lambdify if expression simplified to scalar
            if y_grid.ndim == 0:
                y_grid = np.full_like(x_grid, y_grid.item())

            y_grid = y_grid.astype(np.float64)

            # Check matching length
            if y_grid.shape != x_grid.shape:
                return None

            # Safety checks: finite values, non-zero variance
            if not np.all(np.isfinite(y_grid)):
                return None
            if np.var(y_grid) < self.variance_threshold:
                return None

            return x_grid, y_grid
        except Exception:
            return None
