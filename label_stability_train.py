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



class LabelSage(nn.Module):
    def __init__(self,in_feats,out_feats,cls_num,agg:str='mean') -> None:
        super().__init__()
        self.conv = SAGEConv(in_feats, out_feats, agg)
        self.predictor=nn.Linear(out_feats,cls_num)
        self.predictor_once=nn.Linear(out_feats,cls_num+1)
    
    def forward(self,g,feat):
        t=self.conv(g,feat).relu_()
        out=self.predictor(t)
            
        return out

from dgl.data import FraudDataset


dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]

num_classes = dataset.num_classes
# merge label into nodes
# merge node edge feature info

node_labels = hete_g.ndata['label']
fake_node_labels=torch.hstack([node_labels,(2*torch.ones(1,2)).squeeze(0).bool()])
fake_node_labels[-1]=3


train_mask = hete_g.ndata['train_mask'].bool()
fake_train_mask=torch.hstack([train_mask,torch.ones(1,2).squeeze(0).bool()])
valid_mask = hete_g.ndata['val_mask'].bool()
valid_mask=torch.hstack([valid_mask,torch.zeros(1,2).squeeze(0).bool()])
test_mask = hete_g.ndata['test_mask'].bool()
test_mask=torch.hstack([test_mask,torch.zeros(1,2).squeeze(0).bool()])

x_dim=128

graph = dgl.to_homogeneous(hete_g)
label_idx_start=graph.num_nodes()
normal_id=graph.num_nodes()
fraud_id=graph.num_nodes()+1
starter=node_labels
starter[starter==0]=normal_id
starter[starter==1]=fraud_id
distination=torch.tensor(list(range(graph.num_nodes()))).long()
graph.add_nodes(2) 
graph.add_edges(starter,distination)
graph.ndata['x'] = torch.randn(graph.num_nodes(), x_dim)
fake_node_features=graph.ndata['x']
node_features=fake_node_features[:label_idx_start]


def evaluate(model, graph, feat, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph,feat)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)



out_dim=32
EPOCH=300
iterval=50

for i in range(5,100):
    model = LabelSage(x_dim,out_dim,4)
    optimizer = torch.optim.Adam(model.parameters())

    print('start training...')
    for epoch in range(EPOCH):
        model.train()
        
        logits = model(graph,fake_node_features)
        loss = F.cross_entropy(logits[fake_train_mask], fake_node_labels[fake_train_mask])
        label_rep=fake_node_features[label_idx_start:graph.num_nodes()]
        node_rep=fake_node_features[:label_idx_start]
        

        if epoch!=0 and epoch%iterval==0:
            acc = evaluate(model, graph, fake_node_features, fake_node_labels, valid_mask)  
            print(f'epoch {epoch}, loss {loss.item()}, acc {acc}')
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save model if necessary.  Omitted in this example.

    print('testing...')
    with torch.no_grad():
        accs=[]
        for _ in range(10):
            acc = evaluate(model, graph, fake_node_features, fake_node_labels, test_mask)  
            accs.append(acc)
    import numpy as np
    print(f'final test: {np.mean(accs)}, std {np.std(accs)}')

    print('saving label representation')
    label_rep=fake_node_features[label_idx_start:graph.num_nodes()]
    torch.save(label_rep,f'model_data/label_reps/label_rep{i}.pt')
exit()