import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  

# one million timesteps
maxtimesteps = 1000000

# returns computed amplitude
def compute_amplitude(jx, jy):
    # return jy
    return np.sqrt(jx**2 + jy**2)

# returns velocity from amplitude
def compute_velocity(amplitude):
    v = amplitude[1:] - amplitude[:-1]
    # pdb.set_trace()
    return v

# makes bin of given size (bin_size)
def bin_data(amp, neuron_data):
    bin_size = 1000
    nbins = maxtimesteps // bin_size
    amp_splits = np.split(amp, nbins)
    neuron_splits = np.split(neuron_data, nbins, axis=1)
    
    am = [x.sum()/bin_size for x in amp_splits]
    nm = [x.sum(axis=1) for x in neuron_splits]
    # nm = [x.max(axis=1) for x  in neuron_splits]

    amp_final = np.array(am)
    neu_final = np.vstack(nm)
    return amp_final, neu_final

# returns running average of input vector
def compute_running_average(y):
    sz = 2 # size of window 
    y_avg = [np.mean(y[max(0,i-sz):i+1]) for i in range(len(y))]
    return y_avg

# plot test data actual and predicted values
def plot_test_and_true(Y_test, y_pred_test):
    plt.figure(10)
    x = np.arange(len(Y_test))

    # compute running averages for smooth plotting
    y_test_avg = compute_running_average(Y_test)
    y_pred_test_avg = compute_running_average(y_pred_test)

    # plt.plot(Y_test,'r')
    # plt.plot(y_pred_test,'b')

    # plot results on the test data
    title = 'SVM Regression with Polynomial Kernel (degree 2)'
    ylabel = 'Normalized Amplitude'
    xlabel = 'Test data'
    plt.plot(x, y_test_avg, c='r', marker='.', label='true values')
    plt.plot(x, y_pred_test_avg, c='b', marker='.', label='predicted values')
    plt.scatter(x, Y_test, c='r', marker='.', alpha=.3)
    plt.scatter(x, y_pred_test, c='b', marker='.', alpha=.3)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.show()


# load data
# Note: the path is relative here
data_folder = '../np/'
jx_filename = data_folder + 'jx_np.npy'
jy_filename = data_folder + 'jy_np.npy'
neuron_filename = data_folder + 'tss1_binary.npy'

jx = np.load(jx_filename)
jy = np.load(jy_filename)
neuron_data = np.load(neuron_filename)

amp = compute_amplitude(jx, jy)
v = compute_velocity(amp)

# use only a subset of data
amp = amp[:maxtimesteps]
v = v[:maxtimesteps]
# each row represents a neuron 
neuron_data = neuron_data[:, :maxtimesteps]*1.0

amp_final, neu_final = bin_data(amp, neuron_data)
v_final, neu_final = bin_data(v, neuron_data)
# amp_final = v_final

# plt.figure(0)
# plt.plot(amp_final)
# plt.show()

# prepare data for training 
X = neu_final
Y = amp_final/amp_final.max()
# use this for velocity prediction
# Y = v_final/v_final.max()

# pdb.set_trace()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=2)

# train test splits
train_fraction = .75
s = int(train_fraction*X.shape[0])
X_train = X[:s,:]
X_test = X[s:,:]
Y_train = Y[:s]
Y_test = Y[s:]

# scalar transformation of training and test data
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  


# linear models
# reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
# reg = linear_model.Lasso(alpha=.1)
# reg = linear_model.ElasticNet(alpha=0.1, l1_ratio=.7)

# non-linear models
# reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
# reg = SVR(kernel='linear', C=1e3)
reg = SVR(kernel='poly', C=1.0, degree=2, epsilon=.1, tol=1e-3)

# reg = MLPRegressor(
#     hidden_layer_sizes=(32,32),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',
#     learning_rate_init=0.001, power_t=0.5, max_iter=500000, shuffle=True,
#     random_state=0, tol=0.00005, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# reg = svr_poly

reg.fit(X_train, Y_train)
y_pred = reg.predict(X_train)
y_pred_test = reg.predict(X_test)

error = np.linalg.norm(y_pred - Y_train)/len(Y_train)
error_test = np.linalg.norm(y_pred_test - Y_test)/len(Y_test)
print('Training error: %4f'%(error))
print('Test error: %4f'%(error_test))


plot_test_and_true(Y_test, y_pred_test)