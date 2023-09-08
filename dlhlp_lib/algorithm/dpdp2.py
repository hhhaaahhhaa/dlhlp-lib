"""
Torch module implentation of dynamic programming segmentation algorithm supporting batch acceleration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class DPSegmenter(nn.Module):
    """ Template class of dynamic programming for segmentation. """
    def __init__(self, obj, max_segment_length=1e9, greedy_decode=False) -> None:
        super().__init__()
        self.obj = obj
        self.max_segment_length = max_segment_length
        self.debug = False
        self.greedy_decode = greedy_decode
        if self.greedy_decode:
            self.max_segment_length = 1

    def forward(self, reps: torch.FloatTensor, reference: Dict, lengths=None) -> Tuple[List[int], List[int]]:
        """
        Args:
            reps: Representation sequence with shape (B, T, *dim).
            reference: Dictionary containing everything else required for cost function, which results in large design flexibility.

        Return:
            boundaries  :   List of right boundaries of segments.
            label_token :   List of token labels for segments.

        e.g. :
        Assume there are 100 tokens available.
        If  boundaries = [3, 5]
            label_token = [55, 83]

        then segmentation is :
        | z1 z2 z3 | z4 z5 |
        |    55    |   83  |
        """
        B, T, *dims = reps.shape
        costs = torch.zeros(B, T + 1).to(reps.device)  # record the best segmentation costs that ends at timestep t
        prev = torch.zeros(B, T + 1, dtype=torch.long).to(reps.device)  # record starting point of the best last segment
        tokens = torch.zeros(B, T + 1, dtype=torch.long).to(reps.device)  # record token chosen for the best last segment
        if lengths is not None:
            assert(len(lengths) == B), "Size of lengths should be identical to the batch size."
            lengths = [min(l, T) for l in lengths]
        else:
            lengths = [T] * B
        
        reference["rep_table"] = self.create_rep_table(reps)
        for t in range(1, T+1):
            # if t % 100 == 0:
            #     print(f"Step {t}")
            # segment_costs: (B, T+1), value is inf when the second index >= t since previous segment step must < t
            # best_tokens: (B, T+1), the best token indices for each segment
            segment_costs, best_tokens = self.obj(reps, reference, timestep=t, max_len=self.max_segment_length)
            if self.debug:
                print("Seg:", segment_costs)
                print("Tokens:", best_tokens)
                input()
            if self.debug:
                print(costs + segment_costs)
            val, idx = torch.min(costs + segment_costs, dim=1)
            costs[:, t] = val
            prev[:, t] = idx
            if self.debug:
                print(val, idx)
                input()
            tokens[:, t] = torch.gather(best_tokens, 1, idx.view(-1,1)).view(-1)
            if self.debug:
                print("Costs:", costs)
                print("Prevs:", prev)
                print("Tokens:", tokens)
                input()
        
        # Backtrack
        boundaries_list, label_tokens_list = self.backtrack(tokens, prev, lengths)
        if self.greedy_decode:
            boundaries_list, label_tokens_list = self.merge_segments(boundaries_list, label_tokens_list)
        return boundaries_list, label_tokens_list

    def backtrack(self, tokens, prev, lengths=None):
        boundaries_list, label_tokens_list = [], []
        B = tokens.shape[0]
        for b in range(B):
            st = lengths[b]
            boundaries, label_tokens = [], []
            while st != 0:
                boundaries.append(st)
                label_tokens.append(tokens[b][st].item())
                st = prev[b][st].item()
            boundaries.reverse()
            label_tokens.reverse()
            boundaries_list.append(boundaries)
            label_tokens_list.append(label_tokens)
        return boundaries_list, label_tokens_list
    
    def merge_segments(self, boundaries_list, label_tokens_list):
        """ Only used when greedy decoding """
        new_boundaries_list, new_label_tokens_list = [], []
        for (boundaries, label_tokens) in zip(boundaries_list, label_tokens_list):
            t_cur = label_tokens[0]
            new_boundaries, new_label_tokens = [], []
            for i, t in enumerate(label_tokens):
                if t != t_cur:
                    new_boundaries.append(boundaries[i - 1])
                    new_label_tokens.append(t_cur)
                    t_cur = t
            new_boundaries.append(boundaries[-1])
            new_label_tokens.append(label_tokens[-1])
            new_boundaries_list.append(new_boundaries)
            new_label_tokens_list.append(new_label_tokens)
        return new_boundaries_list, new_label_tokens_list

    def create_rep_table(self, reps):
        _, T, *dims = reps.shape
        rep_table = reps.unsqueeze(1)
        expand_shape = [-1] * rep_table.dim()
        expand_shape[1] = T
        return rep_table.expand(*expand_shape)  # B, T, T, *dims


class Objective(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def core(self, reps, rep_table, reference, mask):
        """ Return loss with shape (B, t, K) """
        raise NotImplementedError
    
    def forward(self, reps, reference, timestep, max_len=1e9):
        """
        rep_table is a tensor of size (B, t, t, *dims) where t = min(max_len, timestep)
        mask is a lower triangular mask as below, where 0: unmasked, 1: masked
            1 1 1 1 0
            1 1 1 0 0
            1 1 0 0 0
            1 0 0 0 0
            0 0 0 0 0
        """
        device = reps.device
        B, T, *dims = reps.shape
        costs = torch.ones(B, T + 1).to(device) * float("inf")
        tokens = torch.zeros(B, T + 1, dtype=torch.long).to(device)
        
        # Slice table for this timestep, we do not need the whole table
        t = min(max_len, timestep)
        mask = ~torch.triu(torch.ones(t, t, dtype=torch.bool)).to(device)
        mask = torch.flip(mask, dims=[0])
        rep_table = reference["rep_table"][:, :min(max_len, timestep), max(0, timestep-max_len):timestep]
        
        # Core loss function
        loss = self.core(reps, rep_table, reference, mask)  # B, t, K

        val, idxs = torch.min(loss, dim=2)
        # print(costs[:, timestep-t:timestep].shape)
        costs[:, timestep-t:timestep] = val.flip(dims=[1])  # need to reverse
        tokens[:, timestep-t:timestep] = idxs.flip(dims=[1])  # need to reverse

        return costs, tokens



class DistanceObjective(Objective):
    def __init__(self, lamb=0) -> None:
        super().__init__()
        self.lamb = lamb

    def pen(self, seg_lengths):
        return self.lamb * (1 - seg_lengths) 

    def core(self, reps, rep_table, reference, mask):
        """
        Args:
            reps: Distance array with shape (B, T, T, K)
        Return loss with shape (B, t, K)
        """
        device = reps.device
        t = mask.shape[0]

        x = rep_table
        l2_loss = torch.sum(x * ~mask.view(1, t, t, 1), dim=2)  # B, t, K
        # print(rep_table)
        # print(centers)
        # print(l2_loss, l2_loss.shape)
        seg_lengths = torch.arange(t).float().to(device) + 1
        seg_loss = self.pen(seg_lengths)
        loss = l2_loss + seg_loss.view(1, t, 1)  # B, t, K

        if "unused_ids" in reference:
            loss[:, :, reference["unused_ids"]] = float("inf")

        return loss


class ClusterObjective(Objective):
    """
    Objective from paper:
    Towards unsupervised phone and word segmentation using self-supervised 
    vector-quantized neural networks (https://arxiv.org/abs/2012.07551, INTERSPEECH 2021)
    """
    def __init__(self, lamb=0) -> None:
        super().__init__()
        self.lamb = lamb

    def pen(self, seg_lengths):
        return self.lamb * (1 - seg_lengths) 

    def core(self, reps, rep_table, reference, mask):
        """
        Args:
            reps: Representation with shape (B, T, T, *dim)
        Return loss with shape (B, t, K)
        """
        device = reps.device
        t = mask.shape[0]
        centers = reference["centers"]  # K, dims
        
        x = (rep_table.unsqueeze(3) - centers) ** 2  # B, t, t, K, dim
        x = torch.sum(x, dim=-1)  # B, t, t, K
        l2_loss = torch.sum(x * ~mask.view(1, t, t, 1), dim=2)  # B, t, K
        # print(rep_table)
        # print(centers)
        # print(l2_loss, l2_loss.shape)
        seg_lengths = torch.arange(t).float().to(device) + 1
        seg_loss = self.pen(seg_lengths)
        loss = l2_loss + seg_loss.view(1, t, 1)  # B, t, K

        if "unused_ids" in reference:
            loss[:, :, reference["unused_ids"]] = float("inf")

        return loss


class FSCLOrigObjective(Objective):
    """
    Similar to ClusterObjective but we first average segment features and pass then through codebook before
    matching centers with L2 loss. Latent space is now perfectly matched!
    """
    def __init__(self, lamb=0) -> None:
        super().__init__()
        self.lamb = lamb

    def pen(self, seg_lengths):
        return self.lamb * (1 - seg_lengths) 

    def core(self, reps, rep_table, reference, mask):
        """
        Return loss with shape (B, t, K)
        """
        device = reps.device
        B, T, *dims = reps.shape
        t = mask.shape[0]
        centers = reference["centers"]  # K, d_code
        codebook_attention = reference["codebook_attention"]

        seg_lengths = torch.arange(t).float().to(device) + 1
        seg_loss = self.pen(seg_lengths)
        x = torch.sum(rep_table * ~mask.view(1, t, t, *([1] * len(dims))), dim=2)  # B, t, *dims
        x = x / seg_lengths.view(1, t, *([1] * len(dims)))  # B, t, *dims
        x, _ = codebook_attention(x)  # B, t, d_code
        x = (x.unsqueeze(2) - centers.view(1, 1, *centers.shape)) ** 2  # B, t, K, d_code
        l2_loss = torch.sum(x, dim=-1) * seg_lengths.view(1, t, 1)  # B, t, K
        loss = l2_loss + seg_loss.view(1, t, 1)  # B, t, K

        if "unused_ids" in reference:
            loss[:, :, reference["unused_ids"]] = float("inf")

        return loss


if __name__ == "__main__":
    import random
    random.seed(0)

    obj = ClusterObjective(lamb=0)
    obj2 =FSCLOrigObjective(lamb=0)
    segmenter = DPSegmenter(obj2, max_segment_length=4)

    reps = torch.randn(3, 7, 2)
    centers = torch.FloatTensor(
        [[0.5, 0.5],
         [0, 0],
         [1, 1]]
    )
    reference = {
        "centers": centers,
    }
    a, b = segmenter(reps, reference)
    print(a)
    print(b)

    x = (reps.unsqueeze(2) - centers.view(1, 1, *centers.shape)) ** 2
    print(torch.min(x.sum(-1), dim=2))
