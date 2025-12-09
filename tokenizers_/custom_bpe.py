from typing import Tuple, List, Dict
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from random import choices
from miditok import REMI



class CustomBPE(REMI):
    """
    Custom BPE tokenizer for MIDI data with musical constraints on merges.
    """

    def __init__(self, *args, velocity_threshold: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_threshold = velocity_threshold
        # consonant intervals mod 12: unison, m3, M3, P4, P5, m6, M6, octave
        self._consonant_pc_intervals = {0, 3, 4, 5, 7, 8, 9}

    # ---- small parsing helpers ----
    def learn_bpe_slow(
        self,
        tokens_path: str | Path,
        vocab_size: int,
        out_dir: str | Path | None = None,
        files_lim: int | None = None,
        save_converted_samples: bool = False,
        print_seq_len_variation: bool = True,
        use_velocity: bool | None = True,
        use_rhythm: bool | None = True,
        use_harmony: bool | None = True,
    ):
        """
        Slow BPE with optional musical constraints:
        - use_velocity: enforce velocity consistency
        - use_rhythm: forbid merges that touch Bar tokens
        - use_harmony: enforce consonant pitch intervals

        Pass True/False to toggle each constraint, or leave as None to use
        class-level defaults (currently: all True).
        """
        # resolve flags (you could store defaults on self if you want)
        print("Starting Custom BPE learning with musical constraints...")
        if use_velocity is None:
            use_velocity = True
        if use_rhythm is None:
            use_rhythm = True
        if use_harmony is None:
            use_harmony = True

        print(
            "Custom slow BPE – constraints:"
            f" velocity={use_velocity}, rhythm={use_rhythm}, harmony={use_harmony}"
        )

        if self.is_multi_voc:
            raise ValueError("Multi-vocabulary tokenizers are not compatible with slow BPE.")
        if self.has_bpe and not self.bpe_slow:
            raise ValueError("Tokenizer already trained with fast BPE; cannot retrain with slow BPE.")
        if vocab_size <= len(self.vocab):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be > current vocabulary ({len(self.vocab)})"
            )

        if isinstance(tokens_path, list):
            files_paths = tokens_path
            tokens_path = files_paths[0].parent
        else:
            tokens_path = Path(tokens_path)
            files_paths = list(tokens_path.glob("**/*.json"))

        if not files_paths:
            raise ValueError("BPE learning: no token json files found")

        files_paths_bpe = (
            choices(files_paths, k=files_lim)
            if files_lim is not None and files_lim < len(files_paths)
            else files_paths
        )

        samples, samples_paths = [], []
        original_lengths = []

        # ---- load samples ----
        for file_path in tqdm(files_paths_bpe, desc="Loading token files"):
            file = self.load_tokens(file_path)
            samples.append(file)
            samples_paths.append(file_path.relative_to(tokens_path))
            if self.unique_track:
                original_lengths.append(len(file["ids"]))
            else:
                original_lengths += [len(track) for track in file["ids"]]

        def replace_token_in_seq(
            seq: List[int], succession: Tuple[int, int], new_event: str
        ):
            j = 0
            while j < len(seq) - 1:
                if tuple(seq[j : j + 2]) == succession:
                    seq[j] = self[f"BPE_{new_event}"]
                    del seq[j + 1]
                else:
                    j += 1

        avg_seq_len = [sum(original_lengths) / len(original_lengths)]
        bpe_comb_nb, bpe_comb_means, bpe_comb_max = [], [], []

        pbar = tqdm(
            total=vocab_size - len(self.vocab),
            desc="Learning custom BPE with constraints",
        )

        while len(self.vocab) < vocab_size:
            occurrences: Dict[Tuple[int, int], int] = {}

            # ---------- KEY PART: only count musically valid pairs ----------
            for sample in samples:
                tracks = [sample["ids"]] if self.unique_track else sample["ids"]
                for track in tracks:
                    for i in range(len(track) - 1):
                        pair = (track[i], track[i + 1])
                        if not self._pair_is_musically_valid(
                            pair,
                            use_velocity=use_velocity,
                            use_rhythm=use_rhythm,
                            use_harmony=use_harmony,
                        ):
                            continue
                        occurrences[pair] = occurrences.get(pair, 0) + 1
            # ----------------------------------------------------------------

            if not occurrences:
                print("No more valid pairs under current constraints; stopping early.")
                break

            # Most frequent admissible pair
            most_rec_tok_succession = max(occurrences, key=occurrences.get)

            # Compute prime token decomposition
            prime_tokens_eq: List[int] = []
            for token in most_rec_tok_succession:
                ttype, tval = self._decode_token(token)
                if ttype == "BPE":
                    prime_tokens_eq += self._prime_ids_from_bpe_val(tval)
                else:
                    prime_tokens_eq.append(token)

            final_event_val = (
                "-".join(map(str, most_rec_tok_succession))
                + "."
                + "-".join(map(str, prime_tokens_eq))
            )
            self.add_to_vocab(f"BPE_{final_event_val}")

            # Replace new BPE token in all samples
            for sample in samples:
                if self.unique_track:
                    replace_token_in_seq(
                        sample["ids"], most_rec_tok_succession, final_event_val
                    )
                else:
                    for track in sample["ids"]:
                        replace_token_in_seq(
                            track, most_rec_tok_succession, final_event_val
                        )

            # metrics (optional)
            avg = []
            for sample in samples:
                if self.unique_track:
                    avg.append(len(sample["ids"]))
                else:
                    avg += [len(track) for track in sample["ids"]]
            avg_seq_len.append(float(np.mean(np.array(avg))))

            nb_combs = np.array([len(prime_tokens_eq)])
            if isinstance(bpe_comb_nb, np.ndarray):
                bpe_comb_nb = np.concatenate([bpe_comb_nb, nb_combs])
            else:
                bpe_comb_nb = nb_combs
            bpe_comb_means.append(float(np.mean(bpe_comb_nb)))
            bpe_comb_max.append(int(np.max(bpe_comb_nb)))

            if print_seq_len_variation:
                pbar.set_postfix(
                    {
                        "seq_len_variation": f"{(avg_seq_len[-1] - avg_seq_len[0]) / avg_seq_len[0] * 100:.2f}",
                        "avg_nb_token_combs": f"{bpe_comb_means[-1]:.2f}",
                        "max_nb_token_combs": f"{bpe_comb_max[-1]}",
                    },
                    refresh=False,
                )
            pbar.update(1)

        pbar.close()
        self.has_bpe = True
        self._MIDITokenizer__set_bpe_slow_tokens_successions()  # name-mangled call

        # Save config / samples (unchanged)
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if save_converted_samples:
                for sample, path in zip(samples, samples_paths):
                    self.save_tokens(
                        sample["ids"],
                        Path(out_dir, path).with_suffix(".json"),
                        sample.get("programs", []),
                    )
            self.save_params(out_dir / "config.txt")

        if print_seq_len_variation and len(avg_seq_len) > 1:
            print(
                f"Mean original length: {avg_seq_len[0]:.2f}\n"
                f"Mean length after BPE: {avg_seq_len[-1]:.2f}\n"
                f"Variation: {(avg_seq_len[-1] - avg_seq_len[0]) / avg_seq_len[0] * 100:.2f} %"
            )

        return bpe_comb_means, bpe_comb_max, avg_seq_len

    def _decode_token(self, tid: int) -> Tuple[str, str]:
        """Return (type, value_str) for a base or BPE token id."""
        tok = self[tid]  # MIDITokenizer.__getitem__
        # e.g. "Pitch_60", "Velocity_96", "BPE_10-11.10-20-30"
        type_, val = tok.split("_", 1)
        return type_, val

    def _prime_ids_from_bpe_val(self, bpe_val: str) -> List[int]:
        """
        For a BPE token string like '10-11.10-20-30',
        return the prime token ids [10, 20, 30].
        """
        parts = bpe_val.split(".")
        if len(parts) != 2:
            return []
        prime_str = parts[1]
        return [int(x) for x in prime_str.split("-") if x]

    def _pitch_values_from_token_id(self, tid: int) -> List[int]:
        """
        Get all pitch values (MIDI numbers) encoded in a token id
        (direct Pitch/NoteOn, or inside a BPE token).
        """
        type_, val = self._decode_token(tid)

        # direct pitch-bearing token
        if type_ in {"Pitch", "NoteOn"}:
            return [int(val)]

        # BPE token: dig into prime ids
        if type_ == "BPE":
            prime_ids = self._prime_ids_from_bpe_val(val)
            pitches = []
            for pid in prime_ids:
                ttype, tval = self._decode_token(pid)
                if ttype in {"Pitch", "NoteOn"}:
                    pitches.append(int(tval))
            return pitches

        return []

    def _velocity_values_from_token_id(self, tid: int) -> List[int]:
        """
        Get all velocity values encoded in a token id
        (direct Velocity, or inside a BPE token).
        """
        type_, val = self._decode_token(tid)

        if type_ == "Velocity":
            return [int(val)]

        if type_ == "BPE":
            prime_ids = self._prime_ids_from_bpe_val(val)
            vels = []
            for pid in prime_ids:
                ttype, tval = self._decode_token(pid)
                if ttype == "Velocity":
                    vels.append(int(tval))
            return vels

        return []

    # ---- musical constraints on a *pair* of token ids ----

    def _rhythm_ok(self, pair: Tuple[int, int]) -> bool:
        """
        Rhythm constraint: forbid merges that touch Bar tokens.
        That prevents a BPE unit from spanning a bar boundary.
        """
        for tid in pair:
            ttype, _ = self._decode_token(tid)
            if ttype == "Bar":
                return False
        return True

    def _velocity_ok(self, pair: Tuple[int, int]) -> bool:
        """
        Velocity constraint: any velocities inside the two tokens
        must be within a threshold.
        """
        all_vels: List[int] = []
        for tid in pair:
            all_vels.extend(self._velocity_values_from_token_id(tid))

        if len(all_vels) <= 1:
            # nothing to compare
            return True

        vmin, vmax = min(all_vels), max(all_vels)
        return (vmax - vmin) <= self.velocity_threshold

    def _harmony_ok(self, pair: Tuple[int, int]) -> bool:
        """
        Harmony constraint: all pitch pairs inside the merged token
        must be consonant (mod 12) according to a simple interval set.
        This is a local approximation; it doesn't know full tonal context.
        """
        all_pitches: List[int] = []
        for tid in pair:
            all_pitches.extend(self._pitch_values_from_token_id(tid))

        if len(all_pitches) <= 1:
            return True

        # check every unordered pair of pitches
        for i in range(len(all_pitches)):
            for j in range(i + 1, len(all_pitches)):
                interval = abs(all_pitches[j] - all_pitches[i]) % 12
                if interval not in self._consonant_pc_intervals:
                    return False
        return True

    def _pair_is_musically_valid(
        self,
        pair: Tuple[int, int],
        use_velocity: bool,
        use_rhythm: bool,
        use_harmony: bool,
    ) -> bool:
        """
        Decide whether this pair of token ids is allowed as a BPE candidate
        under the selected constraints.
        """
        if use_rhythm and not self._rhythm_ok(pair):
            return False
        if use_velocity and not self._velocity_ok(pair):
            return False
        if use_harmony and not self._harmony_ok(pair):
            return False
        return True
