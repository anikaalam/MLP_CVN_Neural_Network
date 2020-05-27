import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
#from sklearn import datasets, svm, metrics
#from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import ssl
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pylab as pl
ssl._create_default_https_context = ssl._create_unverified_context
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')



mnist_data = fetch_openml('mnist_784', version=1)
print ("keyes")
print(mnist_data.keys())
print ("keyes")
# The digits dataset
#mnist = datasets.load_digits()
# rescale the data, use the traditional train/test split
#After scaling by 255, the values of X are in between 0 and 1.
#The value of y is an integer number between 0 and 9
#y represents the digit corresponding to each row in X.
#X, y = mnist.data / 255., mnist.target
#X, y = mnist_data['data'], mnist_data['target']
X, y = mnist.data / 255., mnist.target
#X = mnist.data/ 255.
#y = mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
#print('Shape of X:', X.shape, '\n', 'Shape of y:', y.shape)
digit = X[1]
digit_pixels = digit.reshape(28, 28)
print(digit_pixels)
print('Train Data: ', X_train, '\n', 'Test Data:', X_test, '\n','Train label: ', y_train, '\n', 'Test Label: ', y_test)

le = LabelEncoder()

enc = LabelEncoder().fit(y_train)
Y_encode = enc.transform(y_train)
 #print(Y_encode)
# train the NN
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter = 1000, activation = 'logistic', solver='sgd', batch_size = 10, learning_rate_init = 0.1, verbose = 10) # creates a NN with 2 hidden layers
#mlp.fit(X_train, y_train.values.ravel())  # ravel = produces one dimensional array (like cat in matlab)
#mlp.fit(X_train, y_train)
mlp.fit(X_train, Y_encode)
# predict the class on the test data

print("\nTraining set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

predictions = mlp.predict(X_test)  
print('\nPredictions', predictions)
print ('#####confusion_matrix')
print(confusion_matrix(y_test,predictions))  
#print(classification_report(y_test,predictions)) 

print("weights between input and first hidden layer (without bias):")
print(mlp.coefs_[0])
print("\nweights between first hidden and output (without bias):")
print(mlp.coefs_[1])

print()

for i in range(len(mlp.coefs_)):
    number_neurons_in_layer = mlp.coefs_[i].shape[1]
    for j in range(number_neurons_in_layer):
        weights = mlp.coefs_[i][:,j]
        print(i, j, weights, end=", ")
        print()
    print()

print("Bias values for  hidden layer:")
print(mlp.intercepts_[0])
print("\nBias values for output layer:")
print(mlp.intercepts_[1])

#plt.subplot(132)
#plt.imshow(digit_pixels,
#	       cmap = 'gray',             #color map used to specify colors
#           interpolation='nearest'    #algorithm used to blend square colors; with 'nearest' colors will not be blended
 #         )
#plt.subplot(131)
#plt.imshow(a,                         #numpy array generating the image
          # cmap = 'gray',             #color map used to specify colors
         #  interpolation='nearest'    #algorithm used to blend square colors; with 'nearest' colors will not be blended
         # )
#plt.title('Gray color map, no blending', y=1.02, fontsize=12)
#plt.axis('off')
#j = y[2]
#print ('j',j)
# Changing the labels from string to integers
#import numpy as np
#y = y.astype(np.uint8)
#w = y[2]
#print (w)

'''


print("X", X)

digit = X
digit_pixels = digit.reshape(28,28)
#print ('w',w)
#a = np.arange(w).reshape((28,28))
'''
'''
def pixel_mat(row):
    # we're working with train_df so we want to drop the label column
    vec = X.iloc[0:row].values
    # numpy provides the reshape() function to reorganize arrays into specified shapes
    pixel_mat = vec.reshape(28,28)
    return pixel_mat
r = np.random.randint(0,42000)
print('row',pixel_mat(r))



import pandas as pd
import numpy as np
from sklearn import preprocessing  # converts text labels to numeric labels
from sklearn.model_selection import train_test_split  
# splits the data into train and test
from sklearn.preprocessing  import StandardScaler      # scales the data 
from sklearn.neural_network import MLPClassifier      # neural network
from sklearn.metrics import classification_report, confusion_matrix 

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe: use irisdata[x:y] to select rows or irisdata.head() to see the first 5 columns, 
# size or irisdata = irisdata.shape
# find the data types: irisdata.dtypes
# select a column by its name not by position  -> irisdata.Class
irisdata = pd.read_csv('iris.data', names=names) 

X = irisdata.iloc[:, 0:4] 

print('maximum feature values\n\t', X.max())
print('minimum feature values\n\t', X.min())
'''
