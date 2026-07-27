from enum import Enum, auto
from typing import Set, Dict, List


class TokenType(Enum):
    OPERAND = auto()          # x, pi, e, digits
    BINARY_OPERATOR = auto()  # +, -, *, /, ^
    UNARY_FUNCTION = auto()   # sin, cos, tan, exp, log, sqrt, abs
    OPEN_PAREN = auto()       # (
    CLOSE_PAREN = auto()      # )
    SPECIAL = auto()          # <EOS>, <PAD>


# Target mathematical vocabulary
MATH_TOKENS: List[str] = [
    "x", "pi", "e",
    "+", "-", "*", "/", "^",
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
    "(", ")",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".",
    "<EOS>",
]


class MathGrammarSpec:
    """EBNF specification & rules for mathematical expression tokens."""

    OPERANDS: Set[str] = {"x", "pi", "e", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    BINARY_OPS: Set[str] = {"+", "-", "*", "/", "^"}
    UNARY_FUNCS: Set[str] = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs"}

    @classmethod
    def get_token_type(cls, token: str) -> TokenType:
        if token in cls.OPERANDS or token == ".":
            return TokenType.OPERAND
        if token in cls.BINARY_OPS:
            return TokenType.BINARY_OPERATOR
        if token in cls.UNARY_FUNCS:
            return TokenType.UNARY_FUNCTION
        if token == "(":
            return TokenType.OPEN_PAREN
        if token == ")":
            return TokenType.CLOSE_PAREN
        return TokenType.SPECIAL
