from utils import load_mnist
import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Perceptron(x_train, y_train, x_test, y_test, CLASS_POSTIVE, CLASS_NEGATIVE, LOSS="ReLu"):
    '''
    Use Linear Perceptron to excute a binary classification task on MNIST.
    Input:
        x_train, y_train, x_test, y_test: Given by load_mnist() function.
        CLASS_POSITIVE: int in range [0,9], the class to be considered as positive samples.
        CLASS_NEGATIVE: int in range [0,9], different from CLASS_POSITIVE. the class to be considered as negative samples.
        LOSS: option from {"ReLu","Sigmoid"}, The loss function used by perceptron.
    Return:
        accuracy: float, classification accuracy.
    '''
    # Process Data
    x_train = np.concatenate([x_train[y_train==CLASS_POSTIVE],x_train[y_train==CLASS_NEGATIVE]],0)
    x_train = np.concatenate([x_train,np.ones((x_train.shape[0],1))],1)
    y_train = np.concatenate([np.ones((np.sum(y_train==CLASS_POSTIVE),)), -np.ones((np.sum(y_train==CLASS_NEGATIVE),))], 0)
    x_train = (x_train.T*y_train).T
    x_test = np.concatenate([x_test[y_test==CLASS_POSTIVE],x_test[y_test==CLASS_NEGATIVE]],0)
    x_test = np.concatenate([x_test,np.ones((x_test.shape[0],1))],1)
    y_test = np.concatenate([np.ones((np.sum(y_test==CLASS_POSTIVE),)), -np.ones((np.sum(y_test==CLASS_NEGATIVE),))], 0)
    x_test = (x_test.T*y_test).T

    # Training
    w = np.zeros((x_train.shape[1],))
    epoch_num = 100
    learning_rate = 0.1
    for e in range(epoch_num):
        learning_rate = learning_rate/np.power(2,1/epoch_num)
        if LOSS=="ReLu":
            target = np.sum(x_train*w,1)<=0
            if np.mean(target)==0:
                break
            w = w + learning_rate*np.mean(x_train[target],0)
        if LOSS=="Sigmoid":
            loss = Sigmoid(-np.sum(x_train*w,1))
            w = w + learning_rate*np.mean(x_train.T*loss*(1-loss),1)

    #Testing
    accuracy = np.mean(np.sum(x_test*w,1)>0)

    return accuracy

if __name__=="__main__":
    # Load Data
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Test Different Class Pair
    result = np.identity(10)
    for i in range(10):
        for j in range(i+1,10):
            r = Perceptron(x_train, y_train, x_test, y_test, i, j, "ReLu")
            result[i,j] = r
            result[j,i] = r
    average = np.mean([[result[i,j] for j in range(10) if i!=j] for i in range(10)])
    print("Average accuracy: ", average)

    # Plot
    plt.figure()
    plt.imshow(result)
    plt.colorbar()
    plt.xticks([i for i in range(10)],[i for i in range(10)])
    plt.yticks([i for i in range(10)],[i for i in range(10)])
    plt.title("Accuracy")
    plt.savefig("2.jpg")

_debug = np.array([2,3,3])

