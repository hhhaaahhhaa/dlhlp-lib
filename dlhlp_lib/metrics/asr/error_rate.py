import numpy as np
import pickle
from tqdm import tqdm
from typing import Dict
import jiwer

from dlhlp_lib.parsers.Interfaces import BaseDataParser


def segment2duration(segment, fp):
    res = []
    for (s, e) in segment:
        res.append(
            int(
                np.round(e * 1 / fp)
                - np.round(s * 1 / fp)
            )
        )
    return res


def expand(seq, dur):
    assert len(seq) == len(dur)
    res = []
    for (x, d) in zip(seq, dur):
        if d > 0:
            res.extend([x] * d)
    return res


class FERCalculator(object):
    """
    FER calculation, allow phonemes from different sources. Need to pass unify mappings.
    """
    def __init__(self):
        pass

    def exec(self,
            data_parser: BaseDataParser, 
            queries,
            phoneme_featname1: str, segment_featname1: str, 
            phoneme_featname2: str, segment_featname2: str,
            symbol2unify1: Dict, symbol2unify2: Dict,
            fp: float
        ) -> float:
        phn_feat1 = data_parser.get_feature(phoneme_featname1)
        seg_feat1 = data_parser.get_feature(segment_featname1)
        phn_feat2 = data_parser.get_feature(phoneme_featname2)
        seg_feat2 = data_parser.get_feature(segment_featname2)

        n_frames, correct = 0, 0
        n_seg1, n_seg2 = 0, 0
        for query in tqdm(queries):
            phoneme1 = phn_feat1.read_from_query(query).strip().split(" ")
            segment1 = seg_feat1.read_from_query(query)
            phoneme2 = phn_feat2.read_from_query(query).strip().split(" ")
            segment2 = seg_feat2.read_from_query(query)

            n_seg1 += len(phoneme1)
            n_seg2 += len(phoneme2)

            duration1, duration2 = segment2duration(segment1, fp), segment2duration(segment2, fp)
            seq1, seq2 = expand(phoneme1, duration1), expand(phoneme2, duration2)
            total_len = min(sum(duration1), sum(duration2))

            for (x1, x2) in zip(seq1, seq2):
                if symbol2unify1[x1] == symbol2unify2[x2]:
                    correct += 1
            n_frames += total_len
        facc = correct / n_frames
        fer = 1 - facc

        print(f"Segments: {n_seg1}, {n_seg2}.")
        print(f"Frame error rate: 1 - {correct}/{n_frames} = {fer * 100:.2f}%")
        return fer


class PERCalculator(object):
    """
    PER calculation, allow phonemes from different sources. Need to pass unify mappings.
    Note that PER is not symmetric.
    """
    def __init__(self):
        pass

    def exec(self,
            data_parser: BaseDataParser, 
            queries,
            ref_phoneme_featname: str, pred_phoneme_featname: str,
            symbol_ref2unify: Dict, symbol_pred2unify: Dict
        ) -> float:
        ref_phn_feat = data_parser.get_feature(ref_phoneme_featname)
        pred_phn_feat = data_parser.get_feature(pred_phoneme_featname)

        wer_list = []
        for query in tqdm(queries):
            ref_phoneme = ref_phn_feat.read_from_query(query).strip().split(" ")
            pred_phoneme = pred_phn_feat.read_from_query(query).strip().split(" ")

            ref_sentence = " ".join([symbol_ref2unify[p] for p in ref_phoneme])
            pred_sentence = " ".join([symbol_pred2unify[p] for p in pred_phoneme])

            wer_list.append(jiwer.wer(ref_sentence, pred_sentence))
        wer = sum(wer_list) / len(wer_list)

        print(f"Word error rate: {wer * 100:.2f}%")
        return wer
