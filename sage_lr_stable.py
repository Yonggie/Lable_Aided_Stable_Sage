import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

def pprint(text):
    print(f"\033[031m{text}\033[0m \n")

class LabelSage(nn.Module):
    def __init__(self,in_feats,out_feats,cls_num,agg:str='mean') -> None:
        super().__init__()
        self.conv = SAGEConv(in_feats, out_feats, agg)
        self.predictor=nn.Linear(out_feats,cls_num)
        
    
    def forward(self,g,feat):
        t=self.conv(g,feat).relu_()
        out=self.predictor(t)
            
        return out
    
    def embed(self,g,feat):
        t=self.conv(g,feat).relu_()
        
        return t.detach()


from dgl.data import FraudDataset


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
EPOCH=200
iterval=20



dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]

num_classes = dataset.num_classes
# merge label into nodes
# merge node edge feature info

node_labels = hete_g.ndata['label']


train_mask = hete_g.ndata['train_mask'].bool()
valid_mask = hete_g.ndata['val_mask'].bool()
test_mask = hete_g.ndata['test_mask'].bool()

x_dim=128


gaps=[]
for i in range(100):
    graph = dgl.to_homogeneous(hete_g)
    graph.ndata['x'] = torch.randn(graph.num_nodes(), x_dim)
    node_features=graph.ndata['x']


    model = LabelSage(x_dim,out_dim,2)
    optimizer = torch.optim.Adam(model.parameters())

    print('start training...')
    for epoch in range(EPOCH):
        model.train()
        
        logits = model(graph,node_features)
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        

        # if epoch!=0 and epoch%iterval==0:
        #     acc = evaluate(model, graph, node_features, node_labels, valid_mask)  
        #     print(f'epoch {epoch}, loss {loss.item()}, acc {acc}')
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save model if necessary.  Omitted in this example.

    print('testing...')
    with torch.no_grad():
        accs=[]
        for _ in range(10):
            acc = evaluate(model, graph, node_features, node_labels, test_mask)  
            accs.append(acc)

    print(f'final test: {np.mean(accs)}, std {np.std(accs)}')

    # train classifier
    data=model.embed(graph,node_features).numpy()
    y=node_labels.numpy()

    X_train,X_test,y_train,y_test=train_test_split(data,y)
    LR=LogisticRegression()
    LR.fit(X_train,y_train)

    score=LR.score(X_test,y_test)
    print(f'1st training LR score:{score}')

    # 2nd model train
    graph2 = dgl.to_homogeneous(hete_g)
    graph2.ndata['x'] = torch.randn(graph2.num_nodes(), x_dim)
    node_features2=graph2.ndata['x']


    model2 = LabelSage(x_dim,out_dim,2)
    optimizer2 = torch.optim.Adam(model2.parameters())

    print('start training...')
    for epoch in range(EPOCH):
        model2.train()
        
        logits = model2(graph2,node_features2)
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        

        # if epoch!=0 and epoch%iterval==0:
        #     acc = evaluate(model2, graph2, node_features2, node_labels, valid_mask)  
        #     print(f'epoch {epoch}, loss {loss.item()}, acc {acc}')
            
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

    # checking gap between classifier
    score2=LR.score(model2.embed(graph2,node_features2).numpy(),y)
    score1=LR.score(data,y)
    pprint(f'LR score: {score1:.4f},{score2:.4f}')
    gaps.append(abs(score1-score2))


pprint(f'gap: {np.mean(gaps)}({np.std(gaps)})')
# gap: 0.007971884928406663(0.005921622927774612) 