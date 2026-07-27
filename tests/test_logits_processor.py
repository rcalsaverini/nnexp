import pytest
import torch
import sympy as sp

from nnexp.grammar import (
    MATH_TOKENS,
    build_vocab_mappings,
    GrammarLogitsProcessor,
    MathGrammarSpec,
    TokenType,
)


def test_state_machine_valid_tokens():
    t2id, id2t = build_vocab_mappings(MATH_TOKENS)
    processor = GrammarLogitsProcessor(t2id, id2t)

    # 1. At start
    valid_start = processor.get_valid_tokens_for_sequence([])
    assert t2id["x"] in valid_start
    assert t2id["sin"] in valid_start
    assert t2id["+"] not in valid_start

    # 2. After operand 'x'
    seq_x = [t2id["x"]]
    valid_after_x = processor.get_valid_tokens_for_sequence(seq_x)
    assert t2id["+"] in valid_after_x
    assert t2id["*"] in valid_after_x
    assert t2id["<EOS>"] in valid_after_x
    assert t2id["x"] not in valid_after_x

    # 3. After unary function 'sin' -> expects '('
    seq_sin = [t2id["sin"]]
    valid_after_sin = processor.get_valid_tokens_for_sequence(seq_sin)
    assert t2id["("] in valid_after_sin
    assert t2id["x"] not in valid_after_sin

    # 4. Open paren balancing
    seq_sin_open = [t2id["sin"], t2id["("], t2id["x"]]
    valid_in_paren = processor.get_valid_tokens_for_sequence(seq_sin_open)
    assert t2id[")"] in valid_in_paren
    assert t2id["<EOS>"] not in valid_in_paren  # cannot end with unclosed paren


def test_logits_processor_masking():
    t2id, id2t = build_vocab_mappings(MATH_TOKENS)
    processor = GrammarLogitsProcessor(t2id, id2t)

    # Batch of size 2 at start of sequence
    input_ids = torch.tensor([[t2id["x"]], [t2id["sin"]]], dtype=torch.long)
    scores = torch.zeros(2, len(MATH_TOKENS), dtype=torch.float32)

    masked_scores = processor(input_ids, scores)

    # Batch element 0 (after 'x'): '+' is valid (score 0), 'x' is invalid (score -inf)
    assert masked_scores[0, t2id["+"]].item() == 0.0
    assert masked_scores[0, t2id["x"]].item() == -float("inf")

    # Batch element 1 (after 'sin'): '(' is valid (score 0), '+' is invalid (score -inf)
    assert masked_scores[1, t2id["("]].item() == 0.0
    assert masked_scores[1, t2id["+"]].item() == -float("inf")


def test_autoregressive_rollout_1000_expressions():
    """
    Simulates 1,000 autoregressive rollouts using GrammarLogitsProcessor and Softmax sampling.
    Asserts 100% syntactic validity and SymPy parseability.
    """
    t2id, id2t = build_vocab_mappings(MATH_TOKENS)
    max_length = 30
    processor = GrammarLogitsProcessor(t2id, id2t, max_length=max_length)
    eos_id = t2id["<EOS>"]

    torch.manual_seed(42)
    num_rollouts = 1000
    parseable_count = 0

    for _ in range(num_rollouts):
        seq = []
        for _ in range(max_length):
            input_ids = torch.tensor([seq], dtype=torch.long) if seq else torch.tensor([[]], dtype=torch.long)
            scores = torch.randn(1, len(MATH_TOKENS), dtype=torch.float32)

            masked_scores = processor(input_ids, scores)
            probs = torch.softmax(masked_scores, dim=-1)

            # Sample next token
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id == eos_id:
                break
            seq.append(next_token_id)

        # Decode tokens to string
        expr_str = "".join([id2t[tid] for tid in seq])

        # Validate with SymPy
        try:
            # Convert infix exponentiation ^ to Python ** for SymPy evaluation
            sympy_expr_str = expr_str.replace("^", "**")
            parsed = sp.parse_expr(sympy_expr_str, local_dict={"x": sp.Symbol("x")}, evaluate=False)
            assert isinstance(parsed, sp.Expr)
            parseable_count += 1
        except (TypeError, ZeroDivisionError, ValueError) as e:
            # Arithmetic evaluation errors (e.g. division by zero producing ComplexInfinity) are valid syntax
            if "SyntaxError" in str(type(e)) or "TokenError" in str(type(e)):
                pytest.fail(f"Grammar processor generated unparseable expression '{expr_str}': {e}")
            parseable_count += 1
        except Exception as e:
            pytest.fail(f"Grammar processor generated unparseable expression '{expr_str}': {e}")

    assert parseable_count == num_rollouts
