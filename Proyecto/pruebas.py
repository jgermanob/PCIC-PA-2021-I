# Standard scientific Python imports
#import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from hklearn import LogisticRegression
import numpy as np

digits = datasets.load_digits()

# plts, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)
#     plt.show()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)


# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)


model = LogisticRegression(C = 0.3333, maxiter= 20, n_jobs=2)
theta = np.array([-2.,-1.,1., 2.], dtype=np.float64)
X = np.concatenate((np.ones((5,1), dtype=np.float64), np.array(range(1,16), dtype=np.float64).reshape(3, 5).transpose()/10.0), axis=1)
y = np.array([1,0,1,0,1], dtype=np.float64)
args = [X, y, 0.3333]
# m = X.shape[0]
# theta_aux = theta.copy()
# theta_aux[0] = 0 
# C = 0.3333
# print(X)
# print(y)
# print(m)
# print(theta_aux)
#print(np.matmul(X_train,theta))
# print((1/m)*(np.matmul(-y.transpose(),np.log(model.sigmoid(np.matmul(X,theta, dtype=np.float64))), dtype=np.float64) 
# - np.matmul((1-y).transpose(),np.log(1-model.sigmoid(np.matmul(X,theta, dtype=np.float64))), dtype=np.float64))
# + (1/(2*C*m))*(np.matmul(theta_aux.transpose(),theta_aux, dtype=np.float64)))
print(model.cost_func(theta, *args))
print(model.grad_cost_func(theta, *args))

model.fit(X_train, y_train)
p = model.predict(X_train)
print(p.shape)
print(y_train.shape)