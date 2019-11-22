from utils import *
import numpy as np
import matplotlib.pyplot as plt
def PCA_Gaussian_Bayes(K=None):
    # Load Data
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Compute Covariance and its eigen-decomposision 
    x_mean = np.sum(x_train,0)/x_train.shape[0]
    x_cov = np.matmul((x_train-x_mean).T,(x_train-x_mean))/x_train.shape[0]
    eigVals,eigVects=np.linalg.eig(np.mat(x_cov))

    # Use Top-K principle directions as transformation matrix.
    if K is None:
        K = 30
    A = np.array(eigVects[:,:K])
    x_train_new = np.matmul(x_train,A)
    x_test_new = np.matmul(x_test,A)

    # Estimate Gaussian parameters for each class in K-dimensional space
    mu = []
    sigma_inv = []
    IDEPENDENT_COMPONENT = False
    for i in range(10):
        N = x_train_new[y_train==i].shape[0]
        mu.append(np.sum(x_train_new[y_train==i],0)/N)
        if IDEPENDENT_COMPONENT:
            sigma_inv.append(np.identity(K)*np.linalg.inv(np.matmul((x_train_new[y_train==i]-mu[i]).T,(x_train_new[y_train==i]-mu[i]))/N))
        else:
            sigma_inv.append(np.linalg.inv(np.matmul((x_train_new[y_train==i]-mu[i]).T,(x_train_new[y_train==i]-mu[i]))/N))


    # log-posterior for each class
    log_posterior = []
    for i in range(10):
        log_likelyhood = np.sum(np.multiply(np.matmul(x_test_new-mu[i],sigma_inv[i]),(x_test_new-mu[i])),1)-np.log(np.linalg.det(sigma_inv[i]))
        log_prior = np.log(np.sum(y_train==i)/y_train.shape[0])
        log_posterior.append(log_likelyhood-2*log_prior)

    # Classification based on maximum posterior
    result = np.argmin(np.stack(log_posterior,1),1)
    accuracy = [np.mean(result[y_test==i]==y_test[y_test==i]) for i in range(10)]
    mean_accuracy = np.mean(result==y_test)

    return mean_accuracy,accuracy

overall_accuracy = []
class_accuracy = []
for K in range(30):
    oa, ca = PCA_Gaussian_Bayes(K)
    overall_accuracy.append(oa)
    class_accuracy.append(ca)
plt.figure()
for i in range(10):
    plt.plot(np.linspace(1,30,30), np.array(class_accuracy)[:,i],'--')
plt.plot(np.linspace(1,30,30), overall_accuracy, 'k')
plt.legend(['Class '+str(i+1) for i in range(10)]+['Average'])
plt.title('Accuracy-K')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.savefig("1.jpg")

_debug = np.array([2,3,3])