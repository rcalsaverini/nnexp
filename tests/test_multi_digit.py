import pytest
import sympy as sp

from nnexp.grammar.ebnf import build_vocab_mappings
from nnexp.grammar.logits_processor import GrammarLogitsProcessor
from nnexp.grammar.serializer import expression_from_tokens, tokenize_expression

TOKEN_TO_ID, ID_TO_TOKEN = build_vocab_mappings()


def processor(**kwargs):
    kwargs.setdefault("max_stack_depth", 99)
    kwargs.setdefault("max_length", None)
    return GrammarLogitsProcessor(TOKEN_TO_ID, ID_TO_TOKEN, **kwargs)


def allowed(proc, prefix_tokens):
    ids = [TOKEN_TO_ID[t] for t in prefix_tokens]
    return {ID_TO_TOKEN[i] for i in proc.get_valid_tokens_for_sequence(ids)}


def first_rejected(proc, tokens):
    """Returns the first token the FSM refuses, or None if the whole run is legal."""
    ids = [TOKEN_TO_ID[t] for t in tokens]
    for k, tid in enumerate(ids):
        if tid not in proc.get_valid_tokens_for_sequence(ids[:k]):
            return tokens[k]
    return None


def test_digit_may_follow_a_digit():
    assert "5" in allowed(processor(), ["1"])


def test_multi_digit_integer_is_reachable():
    assert first_rejected(processor(), ["2", "5", "8", "<EOS>"]) is None


def test_integer_digits_are_capped():
    proc = processor(max_integer_digits=3)
    assert "4" in allowed(proc, ["1", "2"])
    assert "4" not in allowed(proc, ["1", "2", "3"])
    # The number ends, but the expression can still continue.
    assert {"+", "*", "<EOS>"} <= allowed(proc, ["1", "2", "3"])


def test_leading_zero_is_not_extended():
    proc = processor()
    assert "5" not in allowed(proc, ["0"])
    # A decimal point is still reachable, so 0.5 can be written.
    assert "." in allowed(proc, ["0"])


def test_fraction_digits_are_capped():
    proc = processor(max_fraction_digits=2)
    assert "7" in allowed(proc, ["1", ".", "5"])
    assert "7" not in allowed(proc, ["1", ".", "5", "6"])


def test_a_number_resets_after_an_operator():
    proc = processor(max_integer_digits=2)
    assert "3" not in allowed(proc, ["1", "2"])
    assert "3" in allowed(proc, ["1", "2", "+"])


def test_second_decimal_point_stays_blocked():
    assert "." not in allowed(processor(), ["1", ".", "5"])


@pytest.mark.parametrize(
    "tokens",
    [
        ["1", "5", "*", "x", "<EOS>"],
        ["x", "^", "2", "+", "2", "5", "8", "<EOS>"],
        ["3", "1", "2", "5", "*", "x", "^", "5", "<EOS>"],
        ["0", ".", "5", "*", "x", "<EOS>"],
        ["1", "2", ".", "7", "5", "-", "x", "<EOS>"],
        ["sin", "(", "1", "0", "*", "x", ")", "<EOS>"],
    ],
)
def test_multi_digit_sequences_are_accepted_and_parse(tokens):
    assert first_rejected(processor(), tokens) is None
    expression_from_tokens(tokens)  # must not raise


def test_every_reachable_number_parses():
    """Walk the FSM greedily over digits and confirm the literal is parseable."""
    proc = processor()
    for seed_digit in "123456789":
        tokens = [seed_digit]
        while True:
            options = allowed(proc, tokens)
            nxt = next((d for d in "0123456789" if d in options), None)
            if nxt is None:
                break
            tokens.append(nxt)
        assert sp.Integer(int("".join(tokens))) == expression_from_tokens(tokens)
        assert len(tokens) <= proc.max_integer_digits


def test_generated_dataset_targets_are_reachable():
    """The rejections that motivated this fix were all multi-digit literals."""
    x = sp.Symbol("x")
    proc = processor()
    for expr in [
        15 * sp.Abs(x**2) - sp.E,
        sp.sin(x + 258),
        sp.exp(sp.tan(10 * x)),
        3125 * x**5,
        x**2 + x - sp.pi**2 + sp.sqrt(2) / 81 + 4,
    ]:
        tokens = tokenize_expression(expr, include_eos=True)
        assert first_rejected(proc, tokens) is None, f"{expr} still unreachable"
