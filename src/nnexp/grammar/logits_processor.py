from typing import Dict, List, Set, Tuple, Optional
import torch
from transformers import LogitsProcessor
from .ebnf import MATH_TOKENS, MathGrammarSpec, TokenType, build_vocab_mappings


DIGITS: Set[str] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}


class GrammarLogitsProcessor(LogitsProcessor):
    """
    Hugging Face LogitsProcessor driven by a pushdown state machine
    enforcing EBNF grammar constraints during autoregressive token generation.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        eos_token: str = "<EOS>",
        max_stack_depth: int = 3,
        max_length: Optional[int] = 30,
    ):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.eos_token = eos_token
        self.eos_token_id = token_to_id.get(eos_token, -1)
        self.max_stack_depth = max_stack_depth
        self.max_length = max_length

        self.digit_ids = {self.token_to_id[d] for d in DIGITS if d in self.token_to_id}
        self.dot_id = self.token_to_id.get(".", -1)

    def get_valid_tokens_for_sequence(self, token_ids: List[int]) -> Set[int]:
        """
        Runs the pushdown state machine over generated token sequence.
        Returns the set of valid next token IDs.
        """
        stack: List[str] = []
        last_type: TokenType = TokenType.SPECIAL
        last_str: str = ""
        tokens_seen: int = 0
        in_float: bool = False

        # Process token history to reconstruct pushdown state & floating-point status
        for tid in token_ids:
            if tid not in self.id_to_token:
                continue
            t_str = self.id_to_token[tid]
            if t_str == self.eos_token:
                break

            t_type = MathGrammarSpec.get_token_type(t_str)
            tokens_seen += 1

            if t_type == TokenType.OPEN_PAREN:
                stack.append("(")
            elif t_type == TokenType.CLOSE_PAREN:
                if stack:
                    stack.pop()

            if t_str == ".":
                in_float = True
            elif t_str in DIGITS:
                pass  # maintain float status for multi-digit floats
            else:
                in_float = False  # reset float status on any operator, paren, or non-digit

            last_type = t_type
            last_str = t_str

        # If last token was decimal point '.', ONLY digits are allowed next
        if last_str == ".":
            return self.digit_ids.copy()

        valid_token_ids: Set[int] = set()

        # Determine budget constraint and whether new nested structures can open
        allow_new_depth = len(stack) < self.max_stack_depth

        if self.max_length is not None:
            remaining_steps = self.max_length - tokens_seen

            # Disallow opening new functions or parens if budget is tight
            if remaining_steps <= (len(stack) + 3):
                allow_new_depth = False

            # Force closing parens / EOS when budget is depleted
            if stack and remaining_steps <= (len(stack) + 1):
                if last_type in (TokenType.OPERAND, TokenType.CLOSE_PAREN):
                    if ")" in self.token_to_id:
                        return {self.token_to_id[")"]}
                elif last_type in (TokenType.BINARY_OPERATOR, TokenType.OPEN_PAREN, TokenType.SPECIAL):
                    # Force a simple terminal operand first before closing
                    operand_ids = self._get_ids_by_types({TokenType.OPERAND}) - ({self.dot_id} if self.dot_id != -1 else set())
                    return operand_ids

            elif not stack and remaining_steps <= 2 and last_type in (TokenType.OPERAND, TokenType.CLOSE_PAREN):
                if self.eos_token_id != -1:
                    return {self.eos_token_id}

        # Determine valid next token types based on state
        if last_type in (TokenType.SPECIAL, TokenType.BINARY_OPERATOR, TokenType.OPEN_PAREN):
            valid_types = {TokenType.OPERAND}
            if allow_new_depth:
                valid_types.add(TokenType.OPEN_PAREN)
                valid_types.add(TokenType.UNARY_FUNCTION)

            valid_ids = self._get_ids_by_types(valid_types)
            if self.dot_id in valid_ids:
                valid_ids.remove(self.dot_id)

            # Disallow literal '0' immediately after division operator '/' to prevent division by zero
            if last_str == "/" and "0" in self.token_to_id:
                zero_id = self.token_to_id["0"]
                if zero_id in valid_ids:
                    valid_ids.remove(zero_id)

            valid_token_ids.update(valid_ids)

            # Unary minus allowed at start or after binary op / open paren (except immediately after '-')
            if "-" in self.token_to_id and last_str != "-":
                valid_token_ids.add(self.token_to_id["-"])

        elif last_type == TokenType.UNARY_FUNCTION:
            # Must be open paren after a unary function name like 'sin' -> 'sin('
            if "(" in self.token_to_id:
                valid_token_ids.add(self.token_to_id["("])

        elif last_type in (TokenType.OPERAND, TokenType.CLOSE_PAREN):
            # Allowed next: Binary operators
            valid_types = {TokenType.BINARY_OPERATOR}
            valid_token_ids.update(self._get_ids_by_types(valid_types))

            # If last token was a digit and current number is not already a float, '.' is valid
            if last_str in DIGITS and not in_float and self.dot_id != -1:
                valid_token_ids.add(self.dot_id)

            # Close paren allowed if stack is non-empty
            if stack and ")" in self.token_to_id:
                valid_token_ids.add(self.token_to_id[")"])

            # EOS allowed if stack is empty and expression has operands
            if not stack and tokens_seen > 0 and self.eos_token_id != -1:
                valid_token_ids.add(self.eos_token_id)

        return valid_token_ids

    def _get_ids_by_types(self, token_types: Set[TokenType]) -> Set[int]:
        res = set()
        for t_str, tid in self.token_to_id.items():
            if MathGrammarSpec.get_token_type(t_str) in token_types:
                res.add(tid)
        return res

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modifies raw logit scores in-place, setting scores of illegal next tokens to -inf.
        input_ids shape: [batch_size, seq_len]
        scores shape: [batch_size, vocab_size]
        """
        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            seq = input_ids[b].tolist()
            valid_tokens = self.get_valid_tokens_for_sequence(seq)

            # Mask invalid tokens
            mask = torch.full_like(scores[b], fill_value=-float("inf"))
            if valid_tokens:
                valid_indices = torch.tensor(list(valid_tokens), dtype=torch.long, device=scores.device)
                mask[valid_indices] = scores[b, valid_indices]

            scores[b] = mask

        return scores
