import os
import numpy as np
from scipy.interpolate import interp1d
from multiprocessing import Pool
from tqdm import tqdm

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.audio.features import Energy, LogMelSpectrogram, get_feature_from_wav, get_f0_from_wav
from dlhlp_lib.audio.tools import wav_normalization
from dlhlp_lib.tts_preprocess.utils import get_alignment, representation_average, ImapWrapper


ENERGY_NN = Energy(
    n_fft=AUDIO_CONFIG["stft"]["filter_length"],
    win_length=AUDIO_CONFIG["stft"]["win_length"],
    hop_length=AUDIO_CONFIG["stft"]["hop_length"]
)
MEL_NN = LogMelSpectrogram(
    sample_rate=AUDIO_CONFIG["audio"]["sampling_rate"],
    n_fft=AUDIO_CONFIG["stft"]["filter_length"],
    win_length=AUDIO_CONFIG["stft"]["win_length"],
    hop_length=AUDIO_CONFIG["stft"]["hop_length"],
    n_mels=AUDIO_CONFIG["mel"]["n_mel_channels"],
    pad=(AUDIO_CONFIG["stft"]["filter_length"] - AUDIO_CONFIG["stft"]["hop_length"]) // 2,
    power=1,
    norm="slaney",
    mel_scale="slaney"
)
SILENCE = ["sil", "sp", "spn"]


def textgrid2segment_and_phoneme(
    dataset: BaseDataParser, query,
    textgrid_featname: str,
    segment_featname: str,
    phoneme_featname: str
) -> None:
    textgrid_feat = dataset.get_feature(textgrid_featname)

    segment_feat = dataset.get_feature(segment_featname)
    phoneme_feat = dataset.get_feature(phoneme_featname)

    tier = textgrid_feat.read_from_query(query).get_tier_by_name("phones")
    phones, segments = get_alignment(tier)

    segment_feat.save(segments, query)
    phoneme_feat.save(" ".join(phones), query)


def textgrid2segment_and_phoneme_mp(
    dataset, queries, 
    textgrid_featname: str,
    segment_featname: str,
    phoneme_featname: str,
    n_workers: int=os.cpu_count()-2, chunksize: int=64,
    ignore_errors: bool=False
) -> None:
    print("[textgrid2segment_and_phoneme_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [textgrid_featname] * n,
        [segment_featname] * n,
        [phoneme_featname] * n,
        [ignore_errors] * n
    ))
    
    fail_cnt = 0
    with Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap(ImapWrapper(textgrid2segment_and_phoneme), tasks, chunksize=chunksize), total=n):
            fail_cnt += 1 - res
    print("[textgrid2segment_and_phoneme_mp]: Skipped: ", fail_cnt)


def trim_wav_by_segment(
    dataset: BaseDataParser, query, sr: int,
    wav_featname: str,
    segment_featname: str,
    wav_trim_featname: str
) -> None:
    wav_feat = dataset.get_feature(wav_featname)
    segment_feat = dataset.get_feature(segment_featname)

    wav_trim_feat = dataset.get_feature(wav_trim_featname)

    wav = wav_feat.read_from_query(query)
    segment = segment_feat.read_from_query(query)

    wav_trim_feat.save(wav[int(sr * segment[0][0]) : int(sr * segment[-1][1])], query)


def trim_wav_by_segment_mp(
    dataset: BaseDataParser, queries, sr: int,
    wav_featname: str,
    segment_featname: str,
    wav_trim_featname: str,
    refresh: bool=False,
    n_workers: int=1, chunksize: int=256,
    ignore_errors: bool=False
) -> None:
    print("[trim_wav_by_segment_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries, [sr] * n,
        [wav_featname] * n,
        [segment_featname] * n,
        [wav_trim_featname] * n,
        [ignore_errors] * n
    ))

    fail_cnt = 0
    if n_workers == 1:
        segment_feat = dataset.get_feature(segment_featname)
        segment_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            res = trim_wav_by_segment(
                dataset, queries[i], sr,
                wav_featname,
                segment_featname,
                wav_trim_featname,
                ignore_errors
            )
            fail_cnt += 1 - res
    else:
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(trim_wav_by_segment), tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    print("[trim_wav_by_segment_mp]: Skipped: ", fail_cnt)


def wav_to_mel_energy_pitch(
    dataset: BaseDataParser, query,
    wav_featname: str,
    mel_featname: str,
    energy_featname: str,
    pitch_featname: str,
    interp_pitch_featname: str
) -> None:
    wav_feat = dataset.get_feature(wav_featname)

    mel_feat = dataset.get_feature(mel_featname)
    energy_feat = dataset.get_feature(energy_featname)
    pitch_feat = dataset.get_feature(pitch_featname)
    interp_pitch_feat = dataset.get_feature(interp_pitch_featname)

    wav = wav_normalization(wav_feat.read_from_query(query))

    pitch = get_f0_from_wav(
        wav,
        sample_rate=AUDIO_CONFIG["audio"]["sampling_rate"],
        hop_length=AUDIO_CONFIG["stft"]["hop_length"]
    )  # (time, )
    mel = get_feature_from_wav(
        wav, feature_nn=MEL_NN
    )  # (n_mels, time)
    energy = get_feature_from_wav(
        wav, feature_nn=ENERGY_NN
    )  # (time, )

    # interpolate
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    interp_pitch = interp_fn(np.arange(0, len(pitch)))

    if np.sum(pitch != 0) <= 1:
        raise ValueError("Zero pitch detected")

    mel_feat.save(mel, query)
    energy_feat.save(energy, query)
    pitch_feat.save(pitch, query)
    interp_pitch_feat.save(interp_pitch, query)


def wav_to_mel_energy_pitch_mp(
    dataset: BaseDataParser, queries,
    wav_featname: str,
    mel_featname: str,
    energy_featname: str,
    pitch_featname: str,
    interp_pitch_featname: str,
    n_workers: int=4, chunksize: int=32,
    ignore_errors: bool=False
) -> bool:
    print("[wav_to_mel_energy_pitch_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [wav_featname] * n,
        [mel_featname] * n,
        [energy_featname] * n,
        [pitch_featname] * n,
        [interp_pitch_featname] * n,
        [ignore_errors] * n
    ))

    fail_cnt =0
    if n_workers == 1:
        for i in tqdm(range(n)):
            res = wav_to_mel_energy_pitch(
                dataset, queries[i],
                wav_featname,
                mel_featname,
                energy_featname,
                pitch_featname,
                interp_pitch_featname,
                ignore_errors
            )
            fail_cnt += 1 - res
    else:
        with Pool(processes=n_workers) as pool:
            for i in tqdm(pool.imap(ImapWrapper(wav_to_mel_energy_pitch), tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    print("[wav_to_mel_energy_pitch_mp]: Skipped: ", fail_cnt)


def segment2duration(
    dataset: BaseDataParser, query, inv_frame_period: float,
    segment_featname: str, 
    duration_featname: str
) -> None:
    segment_feat = dataset.get_feature(segment_featname)

    duration_feat = dataset.get_feature(duration_featname)

    segment = segment_feat.read_from_query(query)
    durations = []
    for (s, e) in segment:
        durations.append(int(np.round(e * inv_frame_period) - np.round(s * inv_frame_period)))
    
    duration_feat.save(durations, query)


def segment2duration_mp(
    dataset: BaseDataParser, queries, inv_frame_period: float,
    segment_featname: str, 
    duration_featname: str,
    refresh: bool=False,
    n_workers=1, chunksize=256,
    ignore_errors: bool=False
) -> None:
    print("[segment2duration_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries, [inv_frame_period] * n,
        [segment_featname] * n,
        [duration_featname] * n,
        [ignore_errors] * n
    ))

    fail_cnt = 0
    if n_workers == 1:
        segment_feat = dataset.get_feature(segment_featname)
        segment_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            res = segment2duration(
                dataset, queries[i], inv_frame_period,
                segment_featname,
                duration_featname,
                ignore_errors
            )
            fail_cnt += 1- res
    else:
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(segment2duration), tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    print("[segment2duration_mp]: Skipped: ", fail_cnt)


def duration_avg_pitch_and_energy(
    dataset: BaseDataParser, query,
    duration_featname: str,
    pitch_featname: str,
    energy_featname: str
) -> None:
    duration_feat = dataset.get_feature(duration_featname)
    pitch_feat = dataset.get_feature(pitch_featname)
    energy_feat = dataset.get_feature(energy_featname)

    avg_pitch_feat = dataset.get_feature(f"{duration_featname}_avg_pitch")
    avg_energy_feat = dataset.get_feature(f"{duration_featname}_avg_energy")

    durations = duration_feat.read_from_query(query)
    pitch = pitch_feat.read_from_query(query)
    energy = energy_feat.read_from_query(query)

    avg_pitch, avg_energy = pitch[:], energy[:]
    avg_pitch = representation_average(avg_pitch, durations)
    avg_energy = representation_average(avg_energy, durations)

    avg_pitch_feat.save(avg_pitch, query)
    avg_energy_feat.save(avg_energy, query)


def duration_avg_pitch_and_energy_mp(
    dataset: BaseDataParser, queries,
    duration_featname: str,
    pitch_featname: str,
    energy_featname: str,
    refresh: bool=False,
    n_workers: int=1, chunksize: int=256,
    ignore_errors: bool=False
) -> None:
    print("[duration_avg_pitch_and_energy_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [duration_featname] * n,
        [pitch_featname] * n,
        [energy_featname] * n,
        [ignore_errors] * n,
    ))

    fail_cnt = 0
    if n_workers == 1:
        duration_feat = dataset.get_feature(duration_featname)
        pitch_feat = dataset.get_feature(pitch_featname)
        energy_feat = dataset.get_feature(energy_featname)
        duration_feat.read_all(refresh=refresh)
        pitch_feat.read_all(refresh=refresh)
        energy_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            res = duration_avg_pitch_and_energy(
                dataset, queries[i],
                duration_featname,
                pitch_featname,
                energy_featname,
                ignore_errors
            )
            fail_cnt += 1 - res
    else:
        with Pool(processes=n_workers) as pool:
            for i in tqdm(pool.imap(ImapWrapper(duration_avg_pitch_and_energy), tasks, chunksize=chunksize), total=n):
                fail_cnt += 1- res
        print("[duration_avg_pitch_and_energy_mp]: Skipped: ", fail_cnt)
