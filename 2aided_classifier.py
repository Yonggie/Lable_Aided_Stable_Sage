from sklearn.linear_model import LogisticRegression
import torch
from sklearn.model_selection import train_test_split
import pickle

data=torch.load('model_data/node_reps/node_rep.pt').numpy()
y=torch.load('model_data/label.pt').numpy()

X_train,X_test,y_train,y_test=train_test_split(data,y)
LR=LogisticRegression()
LR.fit(X_train,y_train)

score=LR.score(X_test,y_test)
print(f'score:{score}') # score:0.8579510836452259

with open('model_data/LR.model','wb') as f:
    pickle.dump(LR,f)
