"""
Custom REMI tokenizer subclasses with different BPE merge rules.

Each class uses learn_bpe_slow for full control over merges:
1. RhythmBPE - Blocks merges involving Bar tokens
2. HarmonicBPE - Only allows merges with harmonically valid pitch intervals
3. VelocityBPE - Only allows merges where velocity differences are small
4. CombinedBPE - All three rules combined
"""
from typing import List, Union, Tuple, Dict
from pathlib import Path
from copy import deepcopy
from random import choices
import re

import numpy as np
from tqdm import tqdm
from miditok import REMI


class CustomBPEBase(REMI):
    """
    Base class for custom BPE tokenizers.
    
    Implements learn_bpe_slow with space-concatenated token naming.
    Subclasses should override should_merge() to implement their rules.
    """
    
    # Valid harmonic intervals (semitones mod 12)
    # 0=unison, 3=minor third, 4=major third, 5=perfect fourth, 
    # 7=perfect fifth, 9=major sixth
    VALID_HARMONIC_INTERVALS = {0, 3, 4, 5, 7, 9}
    
    # Maximum allowed velocity difference
    MAX_VELOCITY_DIFF = 30
    
    def should_merge(self, token1: str, token2: str) -> bool:
        """
        Override in subclasses to implement custom merge rules.
        
        :param token1: First token string (may contain spaces if already merged)
        :param token2: Second token string (may contain spaces if already merged)
        :return: True if merge is allowed, False to skip
        """
        return True
    
    def _extract_pitches(self, token_str: str) -> List[int]:
        """
        Extract all pitch values from a token string.
        Token may be a single token or space-concatenated merged tokens.
        
        :param token_str: Token string like "Pitch_60" or "Position_0 Pitch_60 Velocity_64"
        :return: List of pitch values found
        """
        pitches = []
        # Split by space to handle merged tokens
        parts = token_str.split(" ")
        for part in parts:
            if part.startswith("Pitch_"):
                try:
                    pitch = int(part.split("_")[1])
                    if 21 <= pitch <= 108:
                        pitches.append(pitch)
                except (ValueError, IndexError):
                    pass
        return pitches
    
    def _extract_velocities(self, token_str: str) -> List[int]:
        """
        Extract all velocity values from a token string.
        
        :param token_str: Token string like "Velocity_64" or merged tokens
        :return: List of velocity values found
        """
        velocities = []
        parts = token_str.split(" ")
        for part in parts:
            if part.startswith("Velocity_"):
                try:
                    vel = int(part.split("_")[1])
                    velocities.append(vel)
                except (ValueError, IndexError):
                    pass
        return velocities
    
    def _has_bar_token(self, token_str: str) -> bool:
        """Check if token string contains a Bar token."""
        return "Bar_None" in token_str
    
    def _check_harmonic_intervals(self, pitches: List[int]) -> bool:
        """
        Check if all pitch combinations have valid harmonic intervals.
        
        :param pitches: List of pitch values
        :return: True if all intervals are valid, False otherwise
        """
        if len(pitches) < 2:
            return True
        
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                interval = abs(pitches[i] - pitches[j]) % 12
                if interval not in self.VALID_HARMONIC_INTERVALS:
                    return False
        return True
    
    def _check_velocity_difference(self, velocities: List[int]) -> bool:
        """
        Check if max velocity difference is within allowed range.
        
        :param velocities: List of velocity values
        :return: True if max difference <= MAX_VELOCITY_DIFF, False otherwise
        """
        if len(velocities) < 2:
            return True
        
        max_diff = max(velocities) - min(velocities)
        return max_diff <= self.MAX_VELOCITY_DIFF
    
    def create_merge_name(self, token1_str: str, token2_str: str) -> str:
        """
        Create merged token name by space-concatenating the token strings.
        This allows parsing the merged token back into components.
        
        :param token1_str: First token string
        :param token2_str: Second token string
        :return: Space-concatenated merged name
        """
        return f"{token1_str} {token2_str}"
    
    def learn_bpe_slow(
        self,
        tokens_path: Union[Path, str],
        vocab_size: int,
        out_dir: Union[Path, str] = None,
        files_lim: int = None,
        save_converted_samples: bool = False,
        print_seq_len_variation: bool = True,
    ) -> Tuple[List[float], List[int], List[float]]:
        """
        Learn BPE with custom merge rules and space-concatenated naming.
        
        :param tokens_path: Path to directory containing tokenized JSON files.
        :param vocab_size: Target vocabulary size.
        :param out_dir: Output directory to save results.
        :param files_lim: Limit number of files to use for training.
        :param save_converted_samples: If True, save BPE-encoded samples to out_dir.
        :param print_seq_len_variation: If True, print sequence length stats.
        :return: Tuple of (bpe_comb_means, bpe_comb_max, avg_seq_len) metrics.
        """
        assert not self.is_multi_voc, (
            "Multi-vocabulary tokenizers are not compatible with BPE"
        )
        assert not self.has_bpe, (
            "This tokenizer already has BPE trained"
        )
        assert vocab_size > len(self.vocab), (
            f"vocab_size ({vocab_size}) must be > current vocab size ({len(self.vocab)})"
        )
        
        files_paths = list(Path(tokens_path).glob("**/*.json"))
        assert len(files_paths) > 0, (
            f"No token files found in {tokens_path}"
        )
        
        # Optionally limit number of files
        if files_lim is not None and files_lim < len(files_paths):
            files_paths = choices(files_paths, k=files_lim)
        
        # Load samples
        samples = []
        samples_paths = []
        original_lengths = []
        
        print(f"[{self.__class__.__name__}] Loading token files...")
        for file_path in tqdm(files_paths, desc="Loading token files"):
            file = self.load_tokens(file_path)
            samples.append(file)
            samples_paths.append(file_path.relative_to(tokens_path))
            if self.unique_track:
                original_lengths.append(len(file["ids"]))
            else:
                original_lengths.extend([len(track) for track in file["ids"]])
        
        def replace_token_in_seq(seq: List[int], succession: Tuple[int, int], new_token_id: int):
            """Replace all occurrences of a token pair with the new merged token."""
            j = 0
            while j < len(seq) - 1:
                if tuple(seq[j:j + 2]) == succession:
                    seq[j] = new_token_id
                    del seq[j + 1]
                j += 1
        
        # BPE learning loop
        avg_seq_len = [sum(original_lengths) / len(original_lengths)]
        bpe_comb_nb = []
        bpe_comb_means = []
        bpe_comb_max = []
        
        skipped_merges = 0
        
        print(f"[{self.__class__.__name__}] Starting BPE learning...")
        pbar = tqdm(total=vocab_size - len(self.vocab), desc="Learning BPE")
        
        while len(self.vocab) < vocab_size:
            # Count occurrences of successive token pairs
            occurrences: Dict[Tuple[int, int], int] = {}
            for sample in samples:
                tracks = [sample["ids"]] if self.unique_track else sample["ids"]
                for track in tracks:
                    for i in range(len(track) - 1):
                        pair = tuple(track[i:i + 2])
                        occurrences[pair] = occurrences.get(pair, 0) + 1
            
            if not occurrences:
                print(f"[{self.__class__.__name__}] No more token pairs to merge")
                break
            
            # Sort pairs by frequency (most common first)
            sorted_pairs = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
            
            # Find the best valid merge candidate
            most_rec_tok_succession = None
            for pair, count in sorted_pairs:
                token1_str = self[pair[0]]
                token2_str = self[pair[1]]
                
                # Check if this merge is allowed by subclass rules
                if self.should_merge(token1_str, token2_str):
                    most_rec_tok_succession = pair
                    break
                else:
                    skipped_merges += 1
            
            if most_rec_tok_succession is None:
                print(f"[{self.__class__.__name__}] No valid merges remaining (skipped {skipped_merges} invalid merges)")
                break
            
            # Create the merged token name (space-concatenated)
            token1_str = self[most_rec_tok_succession[0]]
            token2_str = self[most_rec_tok_succession[1]]
            new_token_name = self.create_merge_name(token1_str, token2_str)
            
            # Add to vocabulary
            self.add_to_vocab(new_token_name)
            new_token_id = self[new_token_name]
            
            # Replace in all samples
            for sample in samples:
                if self.unique_track:
                    replace_token_in_seq(sample["ids"], most_rec_tok_succession, new_token_id)
                else:
                    for track in sample["ids"]:
                        replace_token_in_seq(track, most_rec_tok_succession, new_token_id)
            
            # Compute metrics
            lengths = []
            for sample in samples:
                if self.unique_track:
                    lengths.append(len(sample["ids"]))
                else:
                    lengths.extend([len(track) for track in sample["ids"]])
            
            # Count number of original tokens in merged token (by counting spaces + 1)
            num_orig_tokens = new_token_name.count(" ") + 1
            
            avg_seq_len.append(np.mean(lengths))
            nb_combs = np.array([num_orig_tokens])
            bpe_comb_nb = (
                np.concatenate([bpe_comb_nb, nb_combs])
                if len(bpe_comb_nb) > 0
                else nb_combs
            )
            bpe_comb_means.append(np.mean(bpe_comb_nb))
            bpe_comb_max.append(int(np.max(bpe_comb_nb)))
            
            pbar.set_postfix({
                "seq_len_var": f"{(avg_seq_len[-1] - avg_seq_len[0]) / avg_seq_len[0] * 100:.2f}%",
                "avg_combs": f"{bpe_comb_means[-1]:.2f}",
                "max_combs": f"{bpe_comb_max[-1]}",
                "skipped": skipped_merges,
            })
            pbar.update(1)
        
        pbar.close()
        self.has_bpe = True
        
        print(f"[{self.__class__.__name__}] BPE complete. Skipped {skipped_merges} invalid merges.")
        
        # Save results
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            if save_converted_samples:
                print(f"[{self.__class__.__name__}] Saving converted samples...")
                for sample, path in tqdm(zip(samples, samples_paths), desc="Saving samples", total=len(samples)):
                    save_path = out_dir / path
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_tokens(
                        sample["ids"],
                        save_path.with_suffix(".json"),
                        sample["programs"],
                    )
            
            self.save_params(out_dir / "config.txt")
        
        if print_seq_len_variation:
            print(f"[{self.__class__.__name__}] Mean original length: {avg_seq_len[0]:.1f}")
            print(f"[{self.__class__.__name__}] Mean length after BPE: {avg_seq_len[-1]:.1f}")
            print(f"[{self.__class__.__name__}] Variation: {(avg_seq_len[-1] - avg_seq_len[0]) / avg_seq_len[0] * 100:.2f}%")
        
        return bpe_comb_means, bpe_comb_max, avg_seq_len


class RhythmBPE(CustomBPEBase):
    """
    BPE tokenizer that blocks merges involving Bar tokens.
    
    This preserves rhythmic structure by keeping Bar_None as separate tokens.
    """
    
    def should_merge(self, token1: str, token2: str) -> bool:
        """Block merges if either token contains Bar_None."""
        if self._has_bar_token(token1) or self._has_bar_token(token2):
            return False
        return True


class HarmonicBPE(CustomBPEBase):
    """
    BPE tokenizer that only allows merges with harmonically valid pitch intervals.
    
    Valid intervals (mod 12): 0 (unison), 3 (minor third), 4 (major third),
    5 (perfect fourth), 7 (perfect fifth), 9 (major sixth)
    """
    
    def should_merge(self, token1: str, token2: str) -> bool:
        """Only allow merges where all pitch combinations have valid harmonic intervals."""
        # Get all pitches from both tokens
        pitches1 = self._extract_pitches(token1)
        pitches2 = self._extract_pitches(token2)
        all_pitches = pitches1 + pitches2
        
        # If no pitches involved, allow merge
        if len(all_pitches) < 2:
            return True
        
        # Check all pitch combinations for valid harmonic intervals
        return self._check_harmonic_intervals(all_pitches)


class VelocityBPE(CustomBPEBase):
    """
    BPE tokenizer that only allows merges where velocity differences are small.
    
    Maximum allowed velocity difference: 30
    """
    
    def should_merge(self, token1: str, token2: str) -> bool:
        """Only allow merges where max velocity difference is <= 30."""
        # Get all velocities from both tokens
        velocities1 = self._extract_velocities(token1)
        velocities2 = self._extract_velocities(token2)
        all_velocities = velocities1 + velocities2
        
        # If fewer than 2 velocities, allow merge
        if len(all_velocities) < 2:
            return True
        
        # Check velocity difference constraint
        return self._check_velocity_difference(all_velocities)


class CombinedBPE(CustomBPEBase):
    """
    BPE tokenizer that combines all three rules:
    1. No merges involving Bar tokens (rhythm preservation)
    2. Only harmonically valid pitch intervals (harmonic coherence)
    3. Small velocity differences only (dynamic consistency)
    """
    
    def should_merge(self, token1: str, token2: str) -> bool:
        """Apply all three merge rules."""
        # Rule 1: No Bar tokens
        if self._has_bar_token(token1) or self._has_bar_token(token2):
            return False
        
        # Rule 2: Harmonic intervals
        pitches1 = self._extract_pitches(token1)
        pitches2 = self._extract_pitches(token2)
        all_pitches = pitches1 + pitches2
        if len(all_pitches) >= 2:
            if not self._check_harmonic_intervals(all_pitches):
                return False
        
        # Rule 3: Velocity difference
        velocities1 = self._extract_velocities(token1)
        velocities2 = self._extract_velocities(token2)
        all_velocities = velocities1 + velocities2
        if len(all_velocities) >= 2:
            if not self._check_velocity_difference(all_velocities):
                return False
        
        return True


# Backward compatibility alias
REMIWithRules = RhythmBPE
