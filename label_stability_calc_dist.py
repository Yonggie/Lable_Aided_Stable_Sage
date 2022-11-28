from itertools import combinations
import torch
import torch.nn.functional as F
reps=[]
for i in range(100):
    reps.append(torch.load(f'model_data/label_reps/label_rep{i}.pt'))

# print(torch.dist(rep0,rep1,p=2))
# print(torch.dist(rep0,rep2,p=2))
# print(torch.dist(rep0,rep3,p=2))
# print(torch.dist(rep0,rep4,p=2))
# print(torch.dist(rep1,rep2,p=2))
# print(torch.dist(rep1,rep3,p=2))
# print(torch.dist(rep1,rep4,p=2))
# print(torch.dist(rep2,rep3,p=2))
# print(torch.dist(rep2,rep4,p=2))
# print(torch.dist(rep3,rep4,p=2))

cos_dists,euclid_dists,dot_dists=[],[],[]
for rep in reps:
    euclid_dists.append(torch.dist(rep[0],rep[1],p=2).item())
    cos_dists.append(F.cosine_similarity(rep[0].unsqueeze(0),rep[1].unsqueeze(0)).item())
    dot_dists.append((rep[0]@rep[1]).item())

import numpy as np
print(f"euclid: mean {np.mean(euclid_dists)}({np.std(euclid_dists)})")
print(f"cos: mean {np.mean(cos_dists)}({np.std(cos_dists)})")
print(f"dot: mean {np.mean(dot_dists)}({np.std(dot_dists)})")
