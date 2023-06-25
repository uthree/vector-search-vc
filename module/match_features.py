import torch
import torch.nn as nn

def match_features(source, reference, k=4):
    # source: [N, 768, Length], reference: [N, 768, Length]
    source = source.transpose(1, 2)
    reference = reference.transpose(1, 2)
    source_norm = torch.norm(source, dim=2, keepdim=True)
    reference_norm = torch.norm(reference, dim=2, keepdim=True)
    cos_sims = torch.bmm((source / source_norm), (reference / reference_norm).transpose(1, 2))
    best = torch.topk(cos_sims, k, dim=2)
    result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
    result = result.transpose(1, 2)
    return result

