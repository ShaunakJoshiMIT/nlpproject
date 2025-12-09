"""
Custom REMI tokenizer subclasses with different BPE merge rules.

Each class uses learn_bpe_slow for full control over merges:
1. RhythmBPE - Blocks merges involving Bar tokens
2. HarmonicBPE - Only allows merges with harmonically valid pitch intervals
3. VelocityBPE - Only allows merges where velocity differences are small
4. CombinedBPE - All three rules combined
"""
from typing import List, Union, Tuple, Dict, Set
from pathlib import Path
from collections import defaultdict
from random import choices
import heapq

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
    
    def decode_bpe(self, seq):
        """
        Decode BPE tokens back to base tokens.
        
        Handles space-concatenated merged tokens by splitting them
        and converting back to base token IDs.
        
        :param seq: TokSequence or list of TokSequences to decode (modified in place)
        """
        from miditok import TokSequence
        
        # Handle list of sequences recursively
        if isinstance(seq, list):
            for s in seq:
                self.decode_bpe(s)
            return
        
        # Skip if not BPE encoded
        if not self.has_bpe:
            return
        if isinstance(seq, TokSequence) and not seq.ids_bpe_encoded:
            return
        
        decoded_ids = []
        for id_ in seq.ids:
            token_str = self[id_]
            
            # Check if this is a merged token (contains space)
            if " " in token_str:
                # Split back into base tokens
                base_tokens = token_str.split(" ")
                for base_tok in base_tokens:
                    if base_tok in self._vocab_base:
                        decoded_ids.append(self._vocab_base[base_tok])
                    else:
                        # Token not found, keep original (shouldn't happen)
                        decoded_ids.append(id_)
            else:
                # Already a base token
                decoded_ids.append(id_)
        
        seq.ids = decoded_ids
        seq.ids_bpe_encoded = False
        seq.tokens = None  # Will be rebuilt by complete_sequence
    
    def apply_bpe(self, seq):
        """
        Apply BPE encoding to a sequence using our space-concatenated tokens.
        
        This overrides the parent's apply_bpe to handle our custom naming scheme
        where merged tokens are named like "Position_0 Pitch_60" instead of 
        "BPE_{id1-id2}.{prime_ids}".
        
        :param seq: TokSequence or list of TokSequences to encode (modified in place)
        """
        from miditok import TokSequence
        
        # Handle list of sequences
        if isinstance(seq, list):
            for s in seq:
                self.apply_bpe(s)
            return
        
        if not self.has_bpe:
            return
        
        # Build succession mapping: {new_token_id: (tok1_id, tok2_id)}
        # by finding tokens with spaces (merged tokens)
        if not hasattr(self, '_bpe_successions_custom') or self._bpe_successions_custom is None:
            self._bpe_successions_custom = {}
            for token_str, token_id in self._vocab_base.items():
                if " " in token_str:
                    # This is a merged token - get the first two parts' IDs
                    # For multi-merge tokens like "A B C", we find the pair that created it
                    parts = token_str.rsplit(" ", 1)  # Split from right to get last merge
                    if len(parts) == 2:
                        part1_str = parts[0]  # Could be "A B" or just "A"
                        part2_str = parts[1]  # The last token "C" or "B"
                        part1_id = self._vocab_base.get(part1_str)
                        part2_id = self._vocab_base.get(part2_str)
                        if part1_id is not None and part2_id is not None:
                            self._bpe_successions_custom[token_id] = (part1_id, part2_id)
        
        # Apply BPE by repeatedly replacing token pairs with merged tokens
        ids = list(seq.ids)  # Make a copy
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(ids) - 1:
                pair = (ids[i], ids[i + 1])
                # Check if this pair can be merged
                for new_id, succession in self._bpe_successions_custom.items():
                    if succession == pair:
                        ids[i] = new_id
                        del ids[i + 1]
                        changed = True
                        break
                else:
                    i += 1
        
        seq.ids = ids
        seq.ids_bpe_encoded = True
    
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
        Optimized BPE learning with custom merge rules and space-concatenated naming.
        
        Uses incremental pair counting for ~10-20x speedup over naive implementation.
        
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
        all_files_paths = files_paths  # Keep original for applying BPE later
        if files_lim is not None and files_lim < len(files_paths):
            files_paths = choices(files_paths, k=files_lim)
        
        # Load samples - flatten all tracks into single lists for faster processing
        print(f"[{self.__class__.__name__}] Loading {len(files_paths)} token files...")
        sequences: List[List[int]] = []  # List of all sequences (tracks)
        samples_metadata = []  # Keep track of which sequences belong to which file
        original_lengths = []
        
        for file_idx, file_path in enumerate(tqdm(files_paths, desc="Loading token files")):
            file = self.load_tokens(file_path)
            rel_path = file_path.relative_to(tokens_path)
            
            if self.unique_track:
                sequences.append(file["ids"])
                samples_metadata.append((file_idx, rel_path, file["programs"], 0))
                original_lengths.append(len(file["ids"]))
            else:
                for track_idx, track in enumerate(file["ids"]):
                    sequences.append(track)
                    samples_metadata.append((file_idx, rel_path, file["programs"], track_idx))
                    original_lengths.append(len(track))
        
        # Initial pair counting
        print(f"[{self.__class__.__name__}] Computing initial pair counts...")
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        # Track where each pair occurs for incremental updates
        pair_locations: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        
        for seq_idx, seq in enumerate(tqdm(sequences, desc="Counting pairs")):
            for pos in range(len(seq) - 1):
                pair = (seq[pos], seq[pos + 1])
                pair_counts[pair] += 1
                pair_locations[pair].add((seq_idx, pos))
        
        # Cache for token strings to avoid repeated lookups
        token_str_cache: Dict[int, str] = {}
        
        def get_token_str(token_id: int) -> str:
            if token_id not in token_str_cache:
                token_str_cache[token_id] = self[token_id]
            return token_str_cache[token_id]
        
        # BPE learning loop
        avg_seq_len = [sum(original_lengths) / len(original_lengths)]
        bpe_comb_nb = []
        bpe_comb_means = []
        bpe_comb_max = []
        skipped_merges = 0
        
        # Set of pairs that failed validation (to avoid re-checking)
        invalid_pairs: Set[Tuple[int, int]] = set()
        
        print(f"[{self.__class__.__name__}] Starting BPE learning...")
        pbar = tqdm(total=vocab_size - len(self.vocab), desc="Learning BPE")
        
        while len(self.vocab) < vocab_size:
            # Find the most frequent valid pair
            best_pair = None
            best_count = 0
            
            # Sort pairs by count (descending) and find first valid one
            for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
                if count <= 0:
                    break
                if pair in invalid_pairs:
                    continue
                    
                token1_str = get_token_str(pair[0])
                token2_str = get_token_str(pair[1])
                
                if self.should_merge(token1_str, token2_str):
                    best_pair = pair
                    best_count = count
                    break
                else:
                    invalid_pairs.add(pair)
                    skipped_merges += 1
            
            if best_pair is None or best_count <= 0:
                print(f"[{self.__class__.__name__}] No valid merges remaining (skipped {skipped_merges} invalid merges)")
                break
            
            # Create the merged token
            token1_str = get_token_str(best_pair[0])
            token2_str = get_token_str(best_pair[1])
            new_token_name = self.create_merge_name(token1_str, token2_str)
            
            # Add to vocabulary
            self.add_to_vocab(new_token_name)
            new_token_id = self[new_token_name]
            token_str_cache[new_token_id] = new_token_name
            
            # Get locations where this pair occurs
            locations = list(pair_locations[best_pair])
            
            # Process merges and update counts incrementally
            # Sort locations by (seq_idx, pos) in reverse order so deletions don't affect earlier positions
            locations.sort(key=lambda x: (x[0], -x[1]))
            
            processed_seqs = set()
            for seq_idx, pos in locations:
                seq = sequences[seq_idx]
                
                # Check if this position is still valid (might have been affected by earlier merge in same seq)
                if pos >= len(seq) - 1:
                    continue
                if (seq[pos], seq[pos + 1]) != best_pair:
                    continue
                
                # Update counts for affected neighboring pairs
                # Remove old pair from left neighbor
                if pos > 0:
                    old_left_pair = (seq[pos - 1], seq[pos])
                    pair_counts[old_left_pair] -= 1
                    pair_locations[old_left_pair].discard((seq_idx, pos - 1))
                
                # Remove old pair from right neighbor
                if pos + 2 < len(seq):
                    old_right_pair = (seq[pos + 1], seq[pos + 2])
                    pair_counts[old_right_pair] -= 1
                    pair_locations[old_right_pair].discard((seq_idx, pos + 1))
                
                # Do the merge
                seq[pos] = new_token_id
                del seq[pos + 1]
                
                # Add new pairs with neighbors
                if pos > 0:
                    new_left_pair = (seq[pos - 1], new_token_id)
                    pair_counts[new_left_pair] += 1
                    pair_locations[new_left_pair].add((seq_idx, pos - 1))
                
                if pos + 1 < len(seq):
                    new_right_pair = (new_token_id, seq[pos + 1])
                    pair_counts[new_right_pair] += 1
                    pair_locations[new_right_pair].add((seq_idx, pos))
                
                processed_seqs.add(seq_idx)
                
                # Update pair_locations for positions after this merge in same sequence
                # (positions shifted by -1)
                # This is handled implicitly by checking pair validity above
            
            # Remove the merged pair from tracking
            del pair_counts[best_pair]
            del pair_locations[best_pair]
            
            # Compute metrics
            total_length = sum(len(seq) for seq in sequences)
            avg_len = total_length / len(sequences)
            
            num_orig_tokens = new_token_name.count(" ") + 1
            
            avg_seq_len.append(avg_len)
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
        
        # Reconstruct samples from sequences for saving
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            if save_converted_samples:
                print(f"[{self.__class__.__name__}] Saving converted samples...")
                
                # Group sequences back by file
                file_sequences: Dict[Path, Dict] = {}
                for seq_idx, (file_idx, rel_path, programs, track_idx) in enumerate(samples_metadata):
                    if rel_path not in file_sequences:
                        file_sequences[rel_path] = {"ids": [], "programs": programs}
                    
                    if self.unique_track:
                        file_sequences[rel_path]["ids"] = sequences[seq_idx]
                    else:
                        # Ensure list is long enough
                        while len(file_sequences[rel_path]["ids"]) <= track_idx:
                            file_sequences[rel_path]["ids"].append([])
                        file_sequences[rel_path]["ids"][track_idx] = sequences[seq_idx]
                
                for rel_path, data in tqdm(file_sequences.items(), desc="Saving samples"):
                    save_path = out_dir / rel_path
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_tokens(
                        data["ids"],
                        save_path.with_suffix(".json"),
                        data["programs"],
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
