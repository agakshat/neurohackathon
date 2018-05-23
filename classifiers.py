from sklearn import tree
import numpy as np
import pdb
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# compute metrices for testing model accuracy
def print_metrics(Y, y_pred):
    # true positive
    tp = ((Y==y_pred)*(Y==1)).sum()
    # false positive 
    fp = ((Y!=y_pred)*(y_pred==1)).sum()
    # false negative
    fn = ((y_pred==0)*(Y==1)).sum()

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = np.sum(y_pred == Y)/len(Y)

    #pdb.set_trace()
    print('Precision: %4f' %(precision))
    print('Recall: %4f' %(recall))
    print('f1_score: %4f' %(f1_score))
    print('accuracy: %4f' %(accuracy))


# concatenate previous time steps to capture temporal correlation in the dataset
def add_previous(X, Y):
    n = 10
    newX = np.zeros((X.shape[0]-n,X.shape[1]*n))
    for i in range(X.shape[0]-n):
        newX[i,:] = X[i:i+n,:].flatten()

    newY = Y[n:]
    return newX, newY


# load data
# note: relative path
data_folder = '../np/'
X_filename = data_folder + 'Bin100X.npy'
Y_filename = data_folder + 'Bin100_y_leading_to_reach.npy'

X = np.load(X_filename)
Y = np.load(Y_filename)

X = X.transpose()
# X = (X>0)*1.0
ntraindata = 10000
X = X[:ntraindata,:]
Y = Y[:ntraindata]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=42)


# decision trees
# clf = tree.DecisionTreeClassifier()
# clf.fit(X,Y)

# random forest
clf = RandomForestClassifier(random_state=0, class_weight='balanced')
clf.fit(X_train, Y_train)

# naive bayes
# clf = GaussianNB()
# clf.fit(X,Y)

# prediction
y_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print('\nTraining results:')
print_metrics(Y_train, y_pred)
print('\nTesting results:')
print_metrics(Y_test, y_test_pred)

# print decision tree
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("bc")
