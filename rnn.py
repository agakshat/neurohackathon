import numpy as np
import torch 
import torch.nn as nn
import pdb
import torch.nn.functional as F

def print_metrics(Y, y_pred):
  tp = ((Y==y_pred)*(Y==1)).sum()
  fp = ((Y!=y_pred)*(y_pred==1)).sum()
  fn = ((y_pred==0)*(Y==1)).sum()
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(y_pred == Y)/len(Y)
  print('Precision: %4f' %(precision))
  print('Recall: %4f' %(recall))
  print('f1_score: %4f' %(f1_score))
  print('accuracy: %4f' %(accuracy))

device=torch.device('cuda')
X = np.load('Bin100X.npy').transpose()
Y = np.load('Bin100_y_leading_to_reach.npy')
X_train = []
batch_size =100
nhid = 512
for i in np.arange(10,len(X)):
	X_train.append(X[i-10:i])
X_train = np.array(X_train)
test_start = 7000
test_end = 10000
X_test = X_train[test_start:test_end]
Y = Y[10:]
Y_test = Y[test_start:test_end]
X_train = X_train[:test_start]
Y = Y[:test_start]
rnn = nn.LSTM(255, nhid, 1, dropout=0.7, batch_first=True)
fc = nn.Linear(nhid,1)
rnn = rnn.to(device)
fc = fc.to(device)
perm = np.random.permutation(X_train.shape[0])
X_train = X_train[perm]
Y=Y[perm]
iterations=0
wt = torch.from_numpy(np.array([1,10]))
optimizer = torch.optim.Adam([{'params':rnn.parameters()},{'params' : fc.parameters()}])
epochs = 1
while iterations<100000:
	iterations +=1
	# if iterations%X_train.shape[0]<batch_size:
	# 	perm = np.random.permutation(X_train.shape[0])
	# 	X_train = X_train[perm]
	# 	Y=Y[perm]
	# 	epochs+=1
	# 	continue
	# trainx = torch.from_numpy(X_train[iterations*batch_size:(iterations+1)*batch_size]).float()
	# trainy = torch.from_numpy(Y[iterations*batch_size:(iterations+1)*batch_size]).float()
	
	mult = iterations%(X_train.shape[0]//batch_size)
	
	trainx = torch.from_numpy(X_train[mult*batch_size:(mult+1)*batch_size]).float()
	trainy = torch.from_numpy(Y[mult*batch_size:(mult+1)*batch_size]).float()
	if X_train[mult*batch_size:(mult+1)*batch_size].shape[0]<batch_size:
		perm = np.random.permutation(X_train.shape[0])
		X_train = X_train[perm]
		Y=Y[perm]
		epochs+=1
		continue

	trainx = trainx.to(device)
	trainy = trainy.to(device)
	wt = trainy + 1
	#pdb.set_trace()
	loss_fn = nn.BCELoss(weight=wt)
	rnn_out, _ = rnn(trainx)
	rnn_out = rnn_out[:,-1]
	pred = F.sigmoid(fc(F.tanh(rnn_out)))
	pred = pred.squeeze(1)
	loss = loss_fn(pred,trainy)
	optimizer.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(rnn.parameters(),0.5)
	nn.utils.clip_grad_norm_(fc.parameters(), 0.5)
	optimizer.step()
	if iterations%1000==0:
		print("iter",iterations, 'loss', loss)
		test_targ=[]
		predictions=[]
		for test_iters in range(X_test.shape[0]//batch_size):
			testx = torch.from_numpy(X_test[test_iters*batch_size:(test_iters+1)*batch_size]).float()
			testy = torch.from_numpy(Y_test[test_iters*batch_size:(test_iters+1)*batch_size]).float()
			testx = testx.to(device)
			testy = testy.to(device)
			rnn_out, _ = rnn(testx)
			rnn_out = rnn_out[:,-1]
			prob = F.sigmoid(fc(F.tanh(rnn_out)))
			prob = prob.squeeze(1)
			pred = (prob>0.5).cpu().numpy()
			predictions.append(pred)
			test_targ.append(testy)
		#pdb.set_trace()
		test_targ=np.concatenate(test_targ)
		predictions = np.concatenate(predictions)
		
		print_metrics(test_targ, predictions)







#pdb.set_trace()