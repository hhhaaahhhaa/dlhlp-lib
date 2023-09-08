from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
import resemblyzer
from typing import Any, List, Callable

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.audio.features import Energy, LogMelSpectrogram
from dlhlp_lib.utils import tool
from . import functional
from .utils import ImapWrapper, representation_average, remove_outlier
from ..parsers.Feature import Feature
from ..parsers.IOObjects import WavIO


def process_tasks_mp(
    tasks: List,
    func: Callable,
    n_workers: int,
    chunksize: int,
    ignore_errors: bool=False
) -> int:
    """
    Fault torlerable multiprocessing wrapper function using multiprocessing.Pool.
    If ignore_errors is true, returns a fail count.
    Note that multiprocess can not pickle local functions, so we wrap them into callable classes with states holded.
    """
    
    n = len(tasks)
    wrapped_func = ImapWrapper(func, ignore_errors=ignore_errors)
    fail_cnt = 0
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(wrapped_func, tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    else:
        for task in tqdm(tasks):
            res = wrapped_func(task)
            fail_cnt += 1 - res
    return fail_cnt


class textgrid2segment_and_phoneme_process(object):
    def __init__(
        self,
        textgrid_feat: Feature,
        segment_feat: Feature,
        phoneme_feat: Feature
    ) -> None:
        self.textgrid_feat = textgrid_feat
        self.segment_feat = segment_feat
        self.phoneme_feat = phoneme_feat

    def __call__(self, query) -> None:
        tier_obj = self.textgrid_feat.read_from_query(query)
        phones, segments = functional.textgrid2segment_and_phoneme(tier_obj)
        self.phoneme_feat.save(" ".join(phones), query)
        self.segment_feat.save(segments, query)


def textgrid2segment_and_phoneme(
    queries: List,
    textgrid_feat: Feature,
    segment_feat: Feature,
    phoneme_feat: Feature,
    n_workers: int=8,
    chunksize: int=256,
    ignore_errors: bool=False
) -> None:

    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks, 
        textgrid2segment_and_phoneme_process(textgrid_feat, segment_feat, phoneme_feat),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[textgrid2segment_and_phoneme]: Skipped: ", fail_cnt)


class trim_wav_by_segment_process(object):
    def __init__(
        self,
        wav_feat: Feature,
        segment_feat: Feature,
        wav_trim_feat: Feature
    ) -> None:
        self.wav_feat = wav_feat
        self.segment_feat = segment_feat
        self.wav_trim_feat = wav_trim_feat

        assert isinstance(wav_feat.io, WavIO)
        self.sr = wav_feat.io._sr

    def __call__(self, query) -> None:
        wav = self.wav_feat.read_from_query(query)
        segment = self.segment_feat.read_from_query(query)
        wav_trim = functional.trim_wav_by_segment(wav, segment, self.sr)
        self.wav_trim_feat.save(wav_trim, query)


def trim_wav_by_segment(
    queries: List,
    wav_feat: Feature,
    segment_feat: Feature,
    wav_trim_feat: Feature,
    n_workers: int=8,
    chunksize: int=256,
    ignore_errors: bool=False
) -> None:
    
    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks,
        trim_wav_by_segment_process(wav_feat, segment_feat, wav_trim_feat),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[trim_wav_by_segment]: Skipped: ", fail_cnt)


class wav_to_mel_process(object):
    def __init__(
        self,
        wav_feat: Feature,
        mel_feat: Feature,
    ) -> None:
        self.wav_feat = wav_feat
        self.mel_feat = mel_feat
        
        self.converter = LogMelSpectrogram(
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

    def __call__(self, query) -> None:
        wav = self.wav_feat.read_from_query(query)
        mel = functional.wav_to_mel(wav, self.converter)
        self.mel_feat.save(mel, query)


def wav_to_mel(
    queries: List,
    wav_feat: Feature,
    mel_feat: Feature,
    n_workers: int=1,
    chunksize: int=64,
    ignore_errors: bool=False
) -> None:
    
    assert n_workers == 1, "Currently do not support multiprocess."
    
    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks,
        wav_to_mel_process(wav_feat, mel_feat),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[wav_to_mel]: Skipped: ", fail_cnt)


class wav_to_energy_process(object):
    def __init__(
        self,
        wav_feat: Feature,
        energy_feat: Feature,
    ) -> None:
        self.wav_feat = wav_feat
        self.energy_feat = energy_feat
        
        self.converter = Energy(
            n_fft=AUDIO_CONFIG["stft"]["filter_length"],
            win_length=AUDIO_CONFIG["stft"]["win_length"],
            hop_length=AUDIO_CONFIG["stft"]["hop_length"]
        )

    def __call__(self, query) -> None:
        wav = self.wav_feat.read_from_query(query)
        energy = functional.wav_to_energy(wav, self.converter)
        self.energy_feat.save(energy, query)


def wav_to_energy(
    queries: List,
    wav_feat: Feature,
    energy_feat: Feature,
    n_workers: int=1,
    chunksize: int=64,
    ignore_errors: bool=False
) -> None:
    
    assert n_workers == 1, "Currently do not support multiprocess."
    
    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks,
        wav_to_energy_process(wav_feat, energy_feat),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[wav_to_energy]: Skipped: ", fail_cnt)


class wav_to_pitch_process(object):
    def __init__(
        self,
        wav_feat: Feature,
        pitch_feat: Feature,
        interp_pitch_feat: Feature,
    ) -> None:
        self.wav_feat = wav_feat
        self.pitch_feat = pitch_feat
        self.interp_pitch_feat = interp_pitch_feat

    def __call__(self, query) -> None:
        wav = self.wav_feat.read_from_query(query)
        pitch, interp_pitch = functional.wav_to_pitch(
            wav, 
            sample_rate=AUDIO_CONFIG["audio"]["sampling_rate"],
            hop_length=AUDIO_CONFIG["stft"]["hop_length"]
        )
        self.pitch_feat.save(pitch, query)
        self.interp_pitch_feat.save(interp_pitch, query)
    

def wav_to_pitch(
    queries: List,
    wav_feat: Feature,
    pitch_feat: Feature,
    interp_pitch_feat: Feature,
    n_workers: int=8,
    chunksize: int=256,
    ignore_errors: bool=False
) -> None:
    
    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks,
        wav_to_pitch_process(wav_feat, pitch_feat, interp_pitch_feat),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[wav_to_pitch]: Skipped: ", fail_cnt)


class segment2duration_process(object):
    def __init__(
        self,
        segment_feat: Feature,
        duration_feat: Feature,
        fp: float
    ) -> None:
        self.segment_feat = segment_feat
        self.duration_feat = duration_feat
        self.fp = fp
    
    def __call__(self, query) -> None:
        segment = self.segment_feat.read_from_query(query)
        duration = tool.segment2duration(segment, self.fp)
        self.duration_feat.save(duration, query)


def segment2duration(
    queries: List,
    segment_feat: Feature,
    duration_feat: Feature,
    fp: float,
    n_workers: int=8,
    chunksize: int=256,
    ignore_errors: bool=False
) -> None:

    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks,
        segment2duration_process(segment_feat, duration_feat, fp),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[segment2duration]: Skipped: ", fail_cnt)


class duration_avg1D_process(object):
    def __init__(
        self,
        duration_feat: Feature,
        input_feat: Feature,
        output_feat: Feature,
        pad: int=0,
    ) -> None:
        self.duration_feat = duration_feat
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.pad = pad
    
    def __call__(self, query) -> None:
        duration = self.duration_feat.read_from_query(query)
        repr = self.input_feat.read_from_query(query)
        res = representation_average(repr, duration, pad=self.pad)
        self.output_feat.save(res, query)


def duration_avg1D(
    queries: List,
    duration_feat: Feature,
    input_feat: Feature,
    output_feat: Feature,
    pad: int=0,
    n_workers: int=8,
    chunksize: int=256,
    ignore_errors: bool=False
) -> None:

    tasks = [(q,) for q in queries]
    fail_cnt = process_tasks_mp(
        tasks,
        duration_avg1D_process(duration_feat, input_feat, output_feat, pad),
        n_workers=n_workers, chunksize=chunksize, ignore_errors=ignore_errors
    )
    print("[duration_avg1D]: Skipped: ", fail_cnt)


def get_stats1D(
    input_feat: Feature,
) -> List[float]:
    input_feat.read_all()
    scaler = StandardScaler()
    all = []
    for k, v in input_feat._data.items():
        vv = remove_outlier(v)
        if len(vv) > 0:
            scaler.partial_fit(vv.reshape((-1, 1)))
        for x in v:
            all.append(x)
    
    stats = [
        float(min(all)),
        float(max(all)),
        float(scaler.mean_[0]),
        float(scaler.scale_[0])
    ]   

    return stats


# TODO:
# def extract_spk_ref_mel_slices_from_wav(
#     dataset: BaseDataParser, query, sr: int,
#     wav_featname: str,
#     ref_featname: str
# ) -> None:
#     wav_feat = dataset.get_feature(wav_featname)

#     ref_feat = dataset.get_feature(ref_featname)

#     wav = wav_feat.read_from_query(query)

#     wav = resemblyzer.preprocess_wav(wav, source_sr=sr)

#     # Compute where to split the utterance into partials and pad the waveform
#     # with zeros if the partial utterances cover a larger range.
#     wav_slices, mel_slices = resemblyzer.VoiceEncoder.compute_partial_slices(
#         len(wav), rate=1.3, min_coverage=0.75
#     )
#     max_wave_length = wav_slices[-1].stop
#     if max_wave_length >= len(wav):
#         wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
#     # Split the utterance into partials and forward them through the model
#     spk_ref_mel = resemblyzer.wav_to_mel_spectrogram(wav)
#     spk_ref_mel_slices = [spk_ref_mel[s] for s in mel_slices]
    
#     ref_feat.save(spk_ref_mel_slices, query)


# def extract_spk_ref_mel_slices_from_wav_mp(
#     dataset: BaseDataParser, queries, sr: int,
#     wav_featname: str,
#     ref_featname: str,
#     n_workers: int=4, chunksize: int=64,
#     ignore_errors: bool=False
# ) -> None:
#     print("[extract_spk_ref_mel_slices_from_wav_mp]:")
#     n = len(queries)
#     tasks = list(zip(
#         [dataset] * n, queries, [sr] * n,
#         [wav_featname] * n,
#         [ref_featname] * n,
#         [ignore_errors] * n,
#     ))

#     fail_cnt = 0
#     if n_workers == 1:
#         for i in tqdm(range(n)):
#             try:
#                 extract_spk_ref_mel_slices_from_wav(
#                     dataset, queries[i], sr,
#                     wav_featname,
#                     ref_featname
#                 )
#             except:
#                 if ignore_errors:
#                     fail_cnt += 1
#                 else:
#                     raise
#     else:
#         with Pool(processes=n_workers) as pool:
#             for res in tqdm(pool.imap(ImapWrapper(extract_spk_ref_mel_slices_from_wav), tasks, chunksize=chunksize), total=n):
#                 fail_cnt += 1 - res
#     print("[extract_spk_ref_mel_slices_from_wav_mp]: Skipped: ", fail_cnt)
