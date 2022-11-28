import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class Sage(nn.Module):
    def __init__(self,in_feats,out_feats,agg:str='mean') -> None:
        super().__init__()
        self.conv = SAGEConv(in_feats, out_feats, agg)
    
    def forward(self,g,feat):
        out=self.conv(g,feat).relu_()

        return out




from dgl.data import FraudDataset

dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]

num_classes = dataset.num_classes
label = hete_g.ndata['label']
node_labels = hete_g.ndata['label']
train_mask = hete_g.ndata['train_mask'].bool()
valid_mask = hete_g.ndata['val_mask'].bool()
test_mask = hete_g.ndata['test_mask'].bool()

x_dim=128
graph = dgl.to_homogeneous(hete_g)
graph.ndata['x'] = torch.randn(graph.num_nodes(), x_dim)
node_features=graph.ndata['x']


def evaluate(model, graph, feat, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph,feat)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


K=3
out_dim=32
EPOCH=5000
iterval=100

model = Sage(x_dim,out_dim)
optimizer = torch.optim.Adam(model.parameters())

print('start training...')
for epoch in range(EPOCH):
    model.train()
    # forward propagation by using all nodes
    logits = model(graph,node_features)
    # compute loss
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    
    # backward propagation
    
    if epoch%iterval==0:
        # compute validation accuracy
        acc = evaluate(model, graph, node_features, node_labels, valid_mask)  
        print(f'epoch {epoch}, loss {loss.item()}, acc {acc}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save model if necessary.  Omitted in this example.

print('testing...')
with torch.no_grad():
    accs=[]
    for _ in range(10):
        acc = evaluate(model, graph, node_features,node_labels, test_mask)  
        accs.append(acc)
import numpy as np
print(f'final test: {np.mean(accs)}, std {np.std(accs)}')
exit()

# PPRGO implementation, packing, testing
# GraphSage implementation, comparison with PPRGO on accuracy, efficiency
# 