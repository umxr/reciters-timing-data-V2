"""
Quran Word Aligner - Local Mac version
Adapted from Quran_Aligner_Colab.ipynb for full-surah audio files.
Uses MPS (Apple Silicon GPU) acceleration when available.

Usage:
    python run_local.py                          # Process all files in audio/
    python run_local.py --audio audio/001.mp3    # Process a single file
    python run_local.py --model base             # Use a smaller model for testing
"""

import json
import os
import re
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import whisper
import Levenshtein


@dataclass
class WordSegment:
    word: str
    start_ms: int
    end_ms: int


@dataclass
class AlignedSpan:
    index_start: int
    index_end: int
    start_ms: int
    end_ms: int


AYAH_COUNTS = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128,
    111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73,
    54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60,
    49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52,
    44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19,
    26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3,
    6, 3, 5, 4, 5, 6,
]


def normalize_arabic(text: str) -> str:
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = text.replace('\u0640', '')
    return text.strip()


def load_quran_text(quran_file: str) -> Dict[int, str]:
    quran_text = {}
    with open(quran_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                surah = int(parts[0])
                ayah = int(parts[1])
                text = parts[2]
                key = surah * 1000 + ayah
                quran_text[key] = text
    return quran_text


def align_words(recognized: List[WordSegment], reference_words: List[str]) -> List[AlignedSpan]:
    if not recognized or not reference_words:
        return []

    rec_normalized = [normalize_arabic(w.word) for w in recognized]
    ref_normalized = [normalize_arabic(w) for w in reference_words]

    n, m = len(rec_normalized), len(ref_normalized)
    INF = float('inf')
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for j in range(1, m + 1):
        dp[0][j] = j * 0.8
    for i in range(1, n + 1):
        dp[i][0] = i * 1.2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            rec_word = rec_normalized[i - 1]
            ref_word = ref_normalized[j - 1]

            if rec_word == ref_word:
                cost = 0
            else:
                ratio = Levenshtein.ratio(rec_word, ref_word)
                if ratio > 0.8:
                    cost = 0.1
                elif ratio > 0.6:
                    cost = 0.4
                elif ratio > 0.4:
                    cost = 0.7
                else:
                    cost = 1.0

            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1.2,
                dp[i][j - 1] + 0.8,
            )

    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            rec_word = rec_normalized[i - 1]
            ref_word = ref_normalized[j - 1]

            if rec_word == ref_word:
                cost = 0
            else:
                ratio = Levenshtein.ratio(rec_word, ref_word)
                if ratio > 0.8:
                    cost = 0.1
                elif ratio > 0.6:
                    cost = 0.4
                elif ratio > 0.4:
                    cost = 0.7
                else:
                    cost = 1.0

            if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 0.001:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue

        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.2)) < 0.001:
            alignment.append((i - 1, None))
            i -= 1
        elif j > 0:
            alignment.append((None, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()

    spans = []
    for rec_idx, aligned_ref_idx in alignment:
        if rec_idx is not None and aligned_ref_idx is not None:
            spans.append(AlignedSpan(
                index_start=aligned_ref_idx,
                index_end=aligned_ref_idx + 1,
                start_ms=recognized[rec_idx].start_ms,
                end_ms=recognized[rec_idx].end_ms,
            ))

    # Fill gaps
    if spans and len(spans) < len(reference_words):
        covered_indices = {s.index_start for s in spans}
        span_by_idx = {s.index_start: s for s in spans}
        filled_spans = []

        for ref_idx in range(len(reference_words)):
            if ref_idx in covered_indices:
                filled_spans.append(span_by_idx[ref_idx])
            else:
                next_span_idx = None
                for idx in range(ref_idx + 1, len(reference_words)):
                    if idx in covered_indices:
                        next_span_idx = idx
                        break

                prev_span = filled_spans[-1] if filled_spans else None
                next_span = span_by_idx.get(next_span_idx) if next_span_idx else None

                if prev_span and next_span:
                    gap_start = prev_span.end_ms
                    gap_end = next_span.start_ms

                    if gap_end > gap_start:
                        gap_words = next_span_idx - ref_idx
                        word_duration = (gap_end - gap_start) // gap_words
                        filled_spans.append(AlignedSpan(
                            index_start=ref_idx,
                            index_end=ref_idx + 1,
                            start_ms=gap_start,
                            end_ms=gap_start + word_duration,
                        ))
                    else:
                        words_in_segment = list(range(ref_idx, next_span_idx + 1))
                        next_duration = next_span.end_ms - next_span.start_ms
                        word_lengths = [len(normalize_arabic(reference_words[i])) for i in words_in_segment]
                        total_length = sum(word_lengths)

                        current_start = next_span.start_ms
                        for i, word_idx in enumerate(words_in_segment):
                            proportion = word_lengths[i] / total_length if total_length > 0 else 1 / len(words_in_segment)
                            word_duration = int(next_duration * proportion)

                            if word_idx == ref_idx:
                                filled_spans.append(AlignedSpan(
                                    index_start=word_idx,
                                    index_end=word_idx + 1,
                                    start_ms=current_start,
                                    end_ms=current_start + word_duration,
                                ))
                            elif word_idx == next_span_idx:
                                span_by_idx[next_span_idx] = AlignedSpan(
                                    index_start=next_span_idx,
                                    index_end=next_span_idx + 1,
                                    start_ms=current_start,
                                    end_ms=next_span.end_ms,
                                )
                            current_start += word_duration
                elif prev_span:
                    avg_duration = max(prev_span.end_ms - prev_span.start_ms, 300)
                    filled_spans.append(AlignedSpan(
                        index_start=ref_idx,
                        index_end=ref_idx + 1,
                        start_ms=prev_span.end_ms,
                        end_ms=prev_span.end_ms + avg_duration,
                    ))
                elif next_span:
                    avg_duration = max(next_span.end_ms - next_span.start_ms, 300)
                    filled_spans.append(AlignedSpan(
                        index_start=ref_idx,
                        index_end=ref_idx + 1,
                        start_ms=max(0, next_span.start_ms - avg_duration),
                        end_ms=next_span.start_ms,
                    ))

        spans = filled_spans

    return spans


def process_full_surah(
    audio_path: str,
    surah_num: int,
    quran_text: Dict[int, str],
    model,
) -> List[dict]:
    """Process a full surah audio file and return per-ayah alignment results."""
    num_ayahs = AYAH_COUNTS[surah_num - 1]

    # Build concatenated reference with ayah boundary tracking
    all_ref_words = []
    ayah_boundaries = []  # (global_start_idx, global_end_idx, surah, ayah)

    for ayah in range(1, num_ayahs + 1):
        key = surah_num * 1000 + ayah
        if key not in quran_text:
            print(f'  Warning: no reference text for {surah_num}:{ayah}')
            continue
        words = quran_text[key].split()
        start_idx = len(all_ref_words)
        all_ref_words.extend(words)
        end_idx = len(all_ref_words)
        ayah_boundaries.append((start_idx, end_idx, surah_num, ayah))

    print(f'  Reference: {len(all_ref_words)} words across {num_ayahs} ayahs')

    # Transcribe full audio
    print(f'  Transcribing {audio_path}...')
    result = model.transcribe(audio_path, language='ar', word_timestamps=True)

    words = []
    for segment in result.get('segments', []):
        for word_info in segment.get('words', []):
            word = word_info.get('word', '').strip()
            if word:
                words.append(WordSegment(
                    word=word,
                    start_ms=int(word_info['start'] * 1000),
                    end_ms=int(word_info['end'] * 1000),
                ))

    print(f'  Whisper recognized {len(words)} words')

    # Align against full concatenated reference
    spans = align_words(words, all_ref_words)
    print(f'  Aligned {len(spans)} word spans')

    # Split spans back into per-ayah results
    results = []
    for start_idx, end_idx, surah, ayah in ayah_boundaries:
        ayah_spans = []
        for span in spans:
            if start_idx <= span.index_start < end_idx:
                # Convert global index to local (within-ayah) index
                ayah_spans.append([
                    span.index_start - start_idx,
                    span.index_end - start_idx,
                    span.start_ms,
                    span.end_ms,
                ])

        results.append({
            'surah': surah,
            'ayah': ayah,
            'segments': ayah_spans,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Quran Word Aligner - Local Mac version')
    parser.add_argument('--audio', type=str, help='Path to a single audio file (e.g. audio/001.mp3)')
    parser.add_argument('--audio-dir', type=str, default='audio', help='Directory containing surah audio files (default: audio/)')
    parser.add_argument('--model', type=str, default='large-v3', help='Whisper model size (tiny, base, small, medium, large-v3)')
    parser.add_argument('--output', type=str, default='output', help='Output directory (default: output/)')
    parser.add_argument('--quran-text', type=str, default='quran-uthmani.txt', help='Path to Quran text file')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Load Quran text
    quran_file = args.quran_text
    if not os.path.exists(quran_file):
        print(f'Quran text not found at {quran_file}')
        print('Downloading from Tanzil...')
        import urllib.request
        urllib.request.urlretrieve(
            'https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt-2&agree=true',
            quran_file,
        )

    quran_text = load_quran_text(quran_file)
    print(f'Loaded {len(quran_text)} ayahs from reference text')

    # Load Whisper model
    import torch
    device = 'cpu'
    if torch.backends.mps.is_available():
        # Whisper doesn't fully support MPS yet for all ops, but transcribe
        # handles device selection internally. We load on CPU and let it work.
        device = 'cpu'
        print('Apple Silicon detected - Whisper will use CPU (MPS not fully supported by Whisper yet)')
    print(f'Loading Whisper {args.model} model...')
    model = whisper.load_model(args.model, device=device)
    print('Model loaded!')

    # Determine which audio files to process
    audio_files = []
    if args.audio:
        audio_files = [args.audio]
    else:
        audio_dir = Path(args.audio_dir)
        if audio_dir.exists():
            for ext in ['*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac']:
                audio_files.extend(sorted(str(f) for f in audio_dir.glob(ext)))
        # Deduplicate: if both 001.mp3 and 001.wav exist, prefer mp3
        seen_stems = {}
        for f in audio_files:
            stem = Path(f).stem
            if stem not in seen_stems:
                seen_stems[stem] = f
        audio_files = sorted(seen_stems.values())

    if not audio_files:
        print('No audio files found. Place surah audio files in audio/ directory.')
        print('Files should be named by surah number: 001.mp3, 002.mp3, etc.')
        sys.exit(1)

    print(f'\nFound {len(audio_files)} audio file(s) to process')

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process each file
    all_results = []
    for audio_path in audio_files:
        stem = Path(audio_path).stem
        # Extract surah number from filename (e.g., "001" -> 1)
        try:
            surah_num = int(stem)
        except ValueError:
            print(f'\nSkipping {audio_path} - cannot determine surah number from filename')
            print('Expected format: 001.mp3 (surah number)')
            continue

        if surah_num < 1 or surah_num > 114:
            print(f'\nSkipping {audio_path} - surah number {surah_num} out of range (1-114)')
            continue

        print(f'\n{"=" * 60}')
        print(f'Processing Surah {surah_num} from {audio_path}')
        print(f'{"=" * 60}')

        results = process_full_surah(audio_path, surah_num, quran_text, model)
        all_results.extend(results)

        # Save per-surah output
        surah_output = os.path.join(args.output, f'surah_{surah_num:03d}.json')
        with open(surah_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'  Saved to {surah_output}')

    # Save combined output
    if all_results:
        combined_output = os.path.join(args.output, 'all_results.json')
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f'\n{"=" * 60}')
        print(f'Done! {len(all_results)} ayahs processed')
        print(f'Combined output: {combined_output}')
        print(f'Per-surah outputs in: {args.output}/')


if __name__ == '__main__':
    main()
