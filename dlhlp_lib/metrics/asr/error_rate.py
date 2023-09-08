import numpy as np
from tqdm import tqdm
from typing import Dict, Union, List, Optional
import jiwer
from jiwer.transformations import wer_default

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.utils.tool import segment2duration, expand


class FERCalculator(object):
    """ 
    FER calculation. 
    Allow phonemes from different sources, need to pass unify mappings. 
    Support confidence score filtering.
    """
    def __init__(self):
        pass

    def exec(self,
            data_parser: BaseDataParser, 
            queries,
            ref_phoneme_featname: str, ref_segment_featname: str,
            pred_phoneme_featname: str, pred_segment_featname: str,
            symbol_ref2unify: Dict, symbol_pred2unify: Dict,
            fp: float,
            confidence_featname: Optional[str]=None, confidence_thresholds: Optional[List[float]]=None,
            avg=False
        ) -> float:
        ref_phn_feat = data_parser.get_feature(ref_phoneme_featname)
        ref_seg_feat = data_parser.get_feature(ref_segment_featname)
        pred_phn_feat = data_parser.get_feature(pred_phoneme_featname)
        pred_seg_feat = data_parser.get_feature(pred_segment_featname)
        ref_phn_feat.read_all()
        ref_seg_feat.read_all()
        pred_phn_feat.read_all()
        pred_seg_feat.read_all()

        if confidence_featname is not None:
            confidence_feat = data_parser.get_feature(confidence_featname)
            confidence_feat.read_all()

        results, confidences = [], []
        fail_cnt = 0
        for query in tqdm(queries):
            try:
                ref_phoneme = ref_phn_feat.read_from_query(query).strip().split(" ")
                ref_segment = ref_seg_feat.read_from_query(query)
                pred_phoneme = pred_phn_feat.read_from_query(query).strip().split(" ")
                pred_segment = pred_seg_feat.read_from_query(query)
            except:
                fail_cnt += 1
                continue
            ref_duration, pred_duration = segment2duration(ref_segment, fp), segment2duration(pred_segment, fp)
            ref_seq, pred_seq = expand(ref_phoneme, ref_duration), expand(pred_phoneme, pred_duration)

            # Padding
            diff = len(pred_seq) - len(ref_seq)
            if diff >= 0:
                pred_seq = pred_seq[:len(ref_seq)]
            else:
                padding = [pred_seq[-1]] * (-diff)
                pred_seq.extend(padding)
            assert len(pred_seq) == len(ref_seq)
            for (x1, x2) in zip(ref_seq, pred_seq):
                results.append(int(symbol_ref2unify[x1] == symbol_pred2unify[x2]))

            # confidence threshold
            if confidence_featname is not None:
                mat = confidence_feat.read_from_query(query)
                assert mat.shape[0] == len(pred_phoneme)
                confidence = np.max(mat, axis=1)
                confidence_seq = expand(confidence.tolist(), pred_duration)

                # sentence level reduction
                if avg:
                    avg = sum(confidence_seq) / len(confidence_seq)
                    confidence_seq = [avg] * len(confidence_seq)

                if diff >= 0:
                    confidence_seq = confidence_seq[:len(ref_seq)]
                else:
                    padding = [confidence_seq[-1]] * (-diff)
                    confidence_seq.extend(padding)
                confidences.extend(confidence_seq)
            else:
                confidences.extend([1.0] * len(ref_seq))
        print("Skipped: ", fail_cnt)

        res = {
            "n_frames": len(results),
        }
        results = np.array(results)
        confidences = np.array(confidences)
        res["correct"] = np.sum(results)
        facc = np.sum(results) / res["n_frames"]
        fer = 1 - facc
        if confidence_thresholds is not None:
            res["results"] = []
            for threshold in confidence_thresholds:
                activated = np.sum(confidences >= threshold)
                matched = np.sum(results[confidences >= threshold])
                res["results"].append((threshold, matched, activated))

        return res


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
            symbol_ref2unify: Dict, symbol_pred2unify: Dict,
            return_dict: bool=False
        ) -> Union[float, Dict]:
        ref_phn_feat = data_parser.get_feature(ref_phoneme_featname)
        pred_phn_feat = data_parser.get_feature(pred_phoneme_featname)
        ref_phn_feat.read_all()
        pred_phn_feat.read_all()

        wer_list = []
        substitutions, insertions, deletions = 0, 0, 0
        for query in tqdm(queries):
            try:
                ref_phoneme = ref_phn_feat.read_from_query(query).strip().split(" ")
                pred_phoneme = pred_phn_feat.read_from_query(query).strip().split(" ")
            except:
                continue

            ref_sentence = " ".join([symbol_ref2unify[p] for p in ref_phoneme])
            pred_sentence = " ".join([symbol_pred2unify[p] for p in pred_phoneme])

            if return_dict:
                measures = jiwer.compute_measures(ref_sentence, pred_sentence, wer_default, wer_default, return_dict=return_dict)
                wer_list.append(measures['wer'])
                substitutions += measures['substitutions']
                insertions += measures['insertions']
                deletions += measures['deletions']
            else:
                wer_list.append(measures)
        wer = sum(wer_list) / len(wer_list)

        print(f"Word error rate: {wer * 100:.2f}%")
        if return_dict:
            return {
                'wer': wer,
                'substitutions': substitutions,
                'insertions': insertions,
                'deletions': deletions
            }
        else:
            return wer
