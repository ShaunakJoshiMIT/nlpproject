"""
REMI tokenizer subclass that prevents Bar tokens from being merged during BPE training.

By adding "Bar" to the special_tokens list before BPE training, the BpeTrainer will
treat Bar_None as a special token that cannot be merged with other tokens.
"""
from typing import Iterable, List, Optional, Tuple

from miditok import REMI  # replace with TSD/Octuple/etc. if needed
from tokenizers.models import BPE
import math
import random

class REMIToken:
    def __init__(self, raw: str, kind: str, value: Optional[str]):
        self.raw = raw
        self.kind = kind
        self.value = value

    def is_time(self) -> bool:
        # adjust to match your dialect
        return self.kind in {"Bar", "Position", "Tempo", "TimeSig"}

    @property
    def is_note(self) -> bool:
        return self.kind in {"Note-On", "Note-Off"}

    @property
    def is_duration(self) -> bool:
        return self.kind == "Duration"

    @property
    def is_special(self) -> bool:
        # e.g. <BOS>, <EOS>, <PAD>, etc.
        return self.raw.startswith("<") and self.raw.endswith(">")

    @property
    def midi_pitch(self) -> Optional[int]:
        if not self.is_note or self.value is None:
            return None
        try:
            return int(self.value)
        except ValueError:
            return None

    @property
    def pitch_class(self) -> Optional[int]:
        p = self.midi_pitch
        return None if p is None else p % 12


def parse_remi_token(tok: str) -> REMIToken:
    if "_" in tok:
        kind, value = tok.split("_", 1)
    else:
        kind, value = tok, None
    return REMIToken(raw=tok, kind=kind, value=value)

def should_block(pair) -> bool:
    """
    Rules:
    - Block merges that involve a Bar token.
    - Block merges that involve a special token.
    - Block Position+Position and Duration+Duration.
    - Block merges between time and non-time tokens.
    - Block merges between notes whose pitch interval is too large.
    """

    # Convert raw byte tokens → string → REMI tokens
    tok1 = parse_remi_token(pair[0].decode("utf-8", errors="ignore"))
    tok2 = parse_remi_token(pair[1].decode("utf-8", errors="ignore"))

    # RULE 1: never merge Bar with anything
    if tok1.kind == "Bar" or tok2.kind == "Bar":
        return True

    # RULE 2: never merge specials (<BOS>, <EOS>, <PAD>, etc.)
    if tok1.is_special or tok2.is_special:
        return True

    # RULE 3: avoid Position+Position and Duration+Duration
    if tok1.kind == "Position" and tok2.kind == "Position":
        return True
    if tok1.is_duration and tok2.is_duration:
        return True

    # RULE 4: don't merge time with non-time (keep scaffold clean)
    if tok1.is_time() != tok2.is_time():
        # one is time, the other is not
        return True

    # RULE 5: block merges between notes with large pitch intervals
    # (e.g. bigger than a perfect fifth = 7 semitones)
    if tok1.is_note and tok2.is_note:
        p1 = tok1.midi_pitch
        p2 = tok2.midi_pitch
        if p1 is not None and p2 is not None:
            interval = abs(p1 - p2)
            if interval > 7:   # block big jumps (e.g. > perfect fifth)
                return True

    # Otherwise allow
    return False



class REMIWithRules(REMI):
    """
    REMI tokenizer that blocks Bar tokens from BPE merges.

    This is achieved by temporarily adding "Bar" to the special_tokens list
    during BPE training, which prevents Bar_None from being merged.
    """

    def learn_bpe(
        self,
        vocab_size: int,
        iterator: Iterable = None,
        tokens_paths: List[Union[Path, str]] = None,
        start_from_empty_voc: bool = False,
        **kwargs,
    ) -> None:
        """
        Learn BPE while blocking Bar tokens from being merged.

        Adds "Bar" to special_tokens temporarily so that Bar_None cannot
        participate in any BPE merges.
        """
        # Store original special tokens
        original_special_tokens = self.special_tokens.copy()

        # Check if Bar is already in special tokens
        bar_already_special = "Bar" in self.special_tokens

        # Add Bar to special tokens if not already there
        if not bar_already_special:
            self.special_tokens.append("Bar")
            print(f"[REMIWithRules] Added 'Bar' to special_tokens: {self.special_tokens}")
        else:
            print(f"[REMIWithRules] 'Bar' already in special_tokens: {self.special_tokens}")

        # Debug: Show what tokens will be protected
        bar_token_name = "Bar_None"
        if hasattr(self, '_vocab_base') and bar_token_name in self._vocab_base:
            bar_id = self._vocab_base[bar_token_name]
            print(f"[REMIWithRules] Bar_None has vocab ID: {bar_id}")

        print(f"[REMIWithRules] Starting BPE training with Bar tokens protected...")

        try:
            # Call parent's learn_bpe - Bar tokens will be treated as special
            super().learn_bpe(
                vocab_size=vocab_size,
                iterator=iterator,
                tokens_paths=tokens_paths,
                start_from_empty_voc=start_from_empty_voc,
                **kwargs,
            )
            print(f"[REMIWithRules] BPE training completed successfully")

            # Verify Bar tokens weren't merged by checking vocab
            if hasattr(self, 'vocab_bpe'):
                bar_in_bpe = [tok for tok in self.vocab_bpe.keys() if 'Bar' in str(tok)]
                print(f"[REMIWithRules] BPE tokens containing 'Bar': {len(bar_in_bpe)}")
                if bar_in_bpe:
                    print(f"[REMIWithRules] Sample Bar BPE tokens: {list(bar_in_bpe)[:5]}")

        finally:
            # Restore original special tokens
            self.special_tokens = original_special_tokens
            print(f"[REMIWithRules] Restored original special_tokens: {self.special_tokens}")

    # Also override train() in case it's called instead of learn_bpe
    def train(
        self,
        vocab_size: int,
        model: str = "BPE",
        iterator: Optional[Iterable] = None,
        files_paths: Optional[Iterable] = None,
        **kwargs,
    ) -> None:
        """
        Train the tokenizer. For BPE, delegates to learn_bpe with Bar protection.
        """
        # MidiTok version compatibility: prefer `train`, fallback to `learn_bpe`.
        print("[REMIWithRules] Starting training with merge filtering")
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

        # Only adjust BPE models.
        # Newer MidiTok (v3+): uses _model.model (HF tokenizers)
        if getattr(self, "_model", None) is not None and hasattr(self._model, "model"):
            if getattr(self._model.model, "type", None) != "BPE":
                return
            print("[REMIWithRules] Filtering BPE merges after training")
            merges = list(getattr(self._model.model, "get_merges", lambda: [])())
            if not merges:
                return
            print(f"[REMIWithRules] Filtering {len(merges)} merges (first 5: {merges[:5]})")
            filtered = [pair for pair in merges if not should_block(tuple(pair))]
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
            print("[REMIWithRules] Filtering BPE merges after training")
            merges = list(getattr(self.bpe_obj, "merges", []))
            # Some MidiTok versions only expose ranks, not merges

            if len(merges) == 0 and hasattr(self.bpe_obj, "bpe_ranks"):
                merges = list(self.bpe_obj.bpe_ranks.keys())
            if not merges:
                return
            print(f"[REMIWithRules] Filtering {len(merges)} merges (first 5: {merges[:5]})")
            filtered = [pair for pair in merges if not should_block(tuple(pair))]
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
