"""
REMI tokenizer subclass that prevents Bar tokens from being merged during BPE training.

By adding "Bar" to the special_tokens list before BPE training, the BpeTrainer will
treat Bar_None as a special token that cannot be merged with other tokens.
"""
from typing import Iterable, List, Optional, Union
from pathlib import Path

from miditok import REMI


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
        if model == "BPE":
            # Use our overridden learn_bpe
            self.learn_bpe(
                vocab_size=vocab_size,
                iterator=iterator,
                tokens_paths=files_paths,
                **kwargs,
            )
        else:
            # For non-BPE models, use parent's train if available
            parent_train = getattr(super(), "train", None)
            if callable(parent_train):
                parent_train(
                    vocab_size=vocab_size,
                    model=model,
                    iterator=iterator,
                    files_paths=files_paths,
                    **kwargs,
                )
            else:
                raise AttributeError(f"Unsupported model type: {model}")
