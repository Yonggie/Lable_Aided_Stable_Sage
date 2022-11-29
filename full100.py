import dgl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dgl.data import FraudDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class LabelSage(nn.Module):
    def __init__(self,in_feats,out_feats,cls_num,agg:str='mean') -> None:
        super().__init__()
        self.conv = SAGEConv(in_feats, out_feats, agg)
        self.predictor=nn.Linear(out_feats,cls_num)
        self.predictor_once=nn.Linear(out_feats,cls_num+1)
    
    def forward(self,g,feat):
        t=self.conv(g,feat).relu_()
        out=self.predictor(t)
            
        return out,t

    def embed(self,g,feat):
        t=self.conv(g,feat).relu_()

        return t.detach()


def pprint(text):
    print(f"\033[031m{text}\033[0m \n")

def evaluate(model, graph, feat, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,_ = model(graph,feat)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def LR_eval(logistic,model,graph,feat,labels):
    score2=logistic.score(model.embed(graph,feat)[:-2].numpy(),labels)
    return score2






dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]
num_classes = dataset.num_classes
node_labels = hete_g.ndata['label']
fake_node_labels=torch.hstack([node_labels,(torch.zeros(1,2)).squeeze(0).bool()])
fake_node_labels[-1]=1


train_mask = hete_g.ndata['train_mask'].bool()
fake_train_mask=torch.hstack([train_mask,torch.ones(1,2).squeeze(0).bool()])
valid_mask = hete_g.ndata['val_mask'].bool()
valid_mask=torch.hstack([valid_mask,torch.zeros(1,2).squeeze(0).bool()])
test_mask = hete_g.ndata['test_mask'].bool()
test_mask=torch.hstack([test_mask,torch.zeros(1,2).squeeze(0).bool()])

x_dim=128


gaps=[]
for i in range(50):
    pprint(f'exp{i+1}')
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





    out_dim=32
    EPOCH=6000
    iterval=1000


    model = LabelSage(x_dim,out_dim,2)
    optimizer = torch.optim.Adam(model.parameters())

    print('start training...')
    for epoch in range(200):
        model.train()
        
        logits,origin_rep = model(graph,fake_node_features)
        loss = F.cross_entropy(logits[fake_train_mask], fake_node_labels[fake_train_mask])
        node_rep_1st=fake_node_features[label_idx_start:graph.num_nodes()]
        node_rep_1st=fake_node_features[:label_idx_start]
        

        # if epoch!=0 and epoch%iterval==0:
        #     acc = evaluate(model, graph, fake_node_features, fake_node_labels, valid_mask)  
        #     print(f'epoch {epoch}, loss {loss.item()}, acc {acc}')
            
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



    # train Logistic regression

    data=model.embed(graph,fake_node_features)[:-2].numpy()
    y=node_labels.numpy()

    X_train,X_test,y_train,y_test=train_test_split(data,y)
    LR=LogisticRegression()
    LR.fit(X_train,y_train)

    score=LR.score(X_test,y_test)
    print(f'1st training LR score:{score}')

    # 2nd training
    graph2 = dgl.to_homogeneous(hete_g)
    label_idx_start=graph2.num_nodes()
    normal_id=graph2.num_nodes()
    fraud_id=graph2.num_nodes()+1
    starter=node_labels
    starter[starter==0]=normal_id
    starter[starter==1]=fraud_id
    distination=torch.tensor(list(range(graph2.num_nodes()))).long()
    graph2.add_nodes(2) 
    graph2.add_edges(starter,distination)

    frozen_label=origin_rep[-2:].detach().requires_grad_(False)
    # print(frozen_label)
    node_features2=torch.randn(graph2.num_nodes(), x_dim)
    graph2.ndata['x'] = node_features2
    fake_node_features=graph2.ndata['x']

    model2 = LabelSage(x_dim,out_dim,2)
    optimizer = torch.optim.Adam(model2.parameters())

    print('start training...')
    for epoch in range(EPOCH):
        model2.train()
        
        logits,reps = model2(graph2,fake_node_features)
        loss1 = F.cross_entropy(logits[fake_train_mask], fake_node_labels[fake_train_mask])
        
        pos_mask=torch.zeros(1,fake_node_features.shape[0]).squeeze(0).bool()
        pos_mask[fake_node_labels==0]=True
        pos_mask[-1]=False
        pos_mask[-2]=False

        neg_mask=torch.zeros(1,fake_node_features.shape[0]).squeeze(0).bool()
        neg_mask[fake_node_labels==1]=True
        neg_mask[-1]=False
        neg_mask[-2]=False
        
        # bug here 
        pos_nodes=reps[pos_mask]
        pos_label=frozen_label[0].expand_as(pos_nodes)

        neg_nodes=reps[neg_mask]
        neg_label=frozen_label[1].expand_as(neg_nodes)
        # loss2 = -(F.cosine_similarity(pos_label,pos_nodes).mean()+F.cosine_similarity(neg_label,neg_nodes).mean()).mean()
        loss2 = -(torch.dist(pos_label,pos_nodes).mean()+torch.dist(neg_label,neg_nodes).mean()).mean()
        loss=0.5*loss1+0.5*loss2
        


        if epoch!=0 and epoch%iterval==0:
            acc=LR.score(model2.embed(graph2,fake_node_features)[:-2].numpy(),y)
            acc_valid = evaluate(model2, graph2, fake_node_features, fake_node_labels, valid_mask)  
            print(f'epoch {epoch}, loss {loss.item():.4f}, acc {acc:.4f}, valid {acc_valid:.4f}')
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


    # print(frozen_label)

    score1=LR.score(data,y)
    score2=LR.score(model2.embed(graph2,fake_node_features)[:-2].numpy(),y)
    pprint(f'LR score: {score1:.4f},{score2:.4f}')
    gap=abs(score1-score2)
    gaps.append(gap)


print('+'*40)
print(f'final: gap {np.mean(gaps):.4f}({np.std(gaps):.4f})')