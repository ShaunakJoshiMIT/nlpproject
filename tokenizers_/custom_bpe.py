"""
Skeleton subclass to override MidiTok BPE training and filter merges.

Swap the base class import to the tokenizer you use (REMI, TSD, etc.).
Implement `should_block` with your own merge-filtering logic.
"""
from typing import Iterable, Optional, Tuple

from miditok import REMI
from tokenizers.models import BPE


BAR_TOKEN_ID = 5  # Bar_None always occupies index 5 in this project setup.


class REMIWithRules(REMI):
    """
    Filters learned BPE merges after MidiTok training.
    If you call `learn_bpe` (deprecated alias) or `train`, the filtered merge list
    replaces the underlying tokenizer model before use.
    """

    def train(
        self,
        vocab_size: int,
        model: str | None = "BPE",
        iterator: Optional[Iterable] = None,
        files_paths: Optional[Iterable] = None,
        **kwargs,
    ) -> None:
        """
        Train then filter BPE merges. Falls back to standard training for non-BPE.
        """
        # MidiTok version compatibility: prefer `train`, fallback to `learn_bpe`.
        parent_train = getattr(super(), "train", None)
        parent_learn_bpe = getattr(super(), "learn_bpe", None)
        if callable(parent_train):
            parent_train(
                vocab_size=vocab_size,
                model=model,
                iterator=iterator,
                files_paths=files_paths,
                **kwargs,
            )
        elif callable(parent_learn_bpe):
            parent_learn_bpe(
                vocab_size,
                tokens_paths=files_paths,
                model=model,
                iterator=iterator,
                **kwargs,
            )
        else:
            raise AttributeError("Parent tokenizer exposes neither train nor learn_bpe")

        def _sym_to_str(sym: str | bytes) -> str:
            return sym.decode("utf-8") if isinstance(sym, bytes) else sym

        # Only adjust BPE models.
        # Newer MidiTok (v3+): uses _model.model (HF tokenizers)
        if getattr(self, "_model", None) is not None and hasattr(self._model, "model"):
            if getattr(self._model.model, "type", None) != "BPE":
                return
            merges = list(getattr(self._model.model, "get_merges", lambda: [])())
            if not merges:
                return
            vocab_lookup = self._model.get_vocab()

            def _is_bar_symbol(sym: str | bytes) -> bool:
                return vocab_lookup.get(_sym_to_str(sym)) == BAR_TOKEN_ID

            filtered = [
                pair for pair in merges if not (_is_bar_symbol(pair[0]) or _is_bar_symbol(pair[1]))
            ]
            if len(filtered) == len(merges):
                return  # nothing to filter

            vocab = self._model.get_vocab()
            cfg = self._model.model  # keep existing BPE settings
            self._model.model = BPE(
                vocab=vocab,
                merges=filtered,
                continuing_subword_prefix=cfg.continuing_subword_prefix,
                end_of_word_suffix=cfg.end_of_word_suffix,
                dropout=cfg.dropout,
            )
            # Refresh decoding helpers after swapping the model.
            if hasattr(self, "_vocab_learned_bytes_to_tokens"):
                self._vocab_learned_bytes_to_tokens = {}
            # Private name in MidiTok v3
            if hasattr(self, "_MusicTokenizer__create_vocab_learned_bytes_to_tokens"):
                self._MusicTokenizer__create_vocab_learned_bytes_to_tokens()
            return

        # Older MidiTok (v2.x): uses bpe_obj with bpe_ranks
        if hasattr(self, "bpe_obj") and hasattr(self.bpe_obj, "bpe_ranks"):
            merges = list(getattr(self.bpe_obj, "merges", []))
            if not merges:
                return
            vocab_lookup = getattr(self.bpe_obj, "vocab", None)
            if not isinstance(vocab_lookup, dict):
                vocab_lookup = getattr(self, "vocab", {})

            def _is_bar_symbol(sym: str | bytes) -> bool:
                return vocab_lookup.get(_sym_to_str(sym)) == BAR_TOKEN_ID

            filtered = [
                pair for pair in merges if not (_is_bar_symbol(pair[0]) or _is_bar_symbol(pair[1]))
            ]
            if len(filtered) == len(merges):
                return
            # Update ranks in-place
            self.bpe_obj.bpe_ranks = {pair: i for i, pair in enumerate(filtered)}
            # Keep merges attribute if present
            try:
                self.bpe_obj.merges = filtered
            except Exception:
                pass
            return

    # Optional: keep the old `learn_bpe` alias for code that still calls it.
    def learn_bpe(
        self,
        vocab_size: int,
        tokens_paths: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:

        return self.train(vocab_size, model="BPE", files_paths=tokens_paths, **kwargs)
