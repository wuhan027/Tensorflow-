'''
Python Basics with Numpy    https://blog.csdn.net/Koala_Tree/article/details/78057033

- np.exp(x) works for any np.array x and applies the exponential function to every coordinate
- the sigmoid function and its gradient
- image2vector is commonly used in deep learning
- np.reshape is widely used. In the future, you’ll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs.
- numpy has efficient built-in functions
- broadcasting is extremely useful

- Vectorization is very important in deep learning. It provides computational efficiency and clarity.
- You have reviewed the L1 and L2 loss.
- You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc…
'''
import math
import numpy as np
import time


def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1.0/(1+math.exp(-x))

    return s


# GRADED FUNCTION: sigmoid
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1.0 / (1 + np.exp(-x))
    ### END CODE HERE ###

    return s


# x = np.array([1, 2, 3])
# print(sigmoid(x))


# GRADED FUNCTION: sigmoid_derivative
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    sig = sigmoid(x)
    ds = sig*(1-sig)

    return ds


# x = np.array([0.5, 1, 2, 3])
# print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))


# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))

    return v


# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],
#
#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],
#
#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])
#
# print ("image2vector(image) = " + str(image2vector(image)))


# GRADED FUNCTION: normalizeRows
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm

    return x


# x = np.array([
#     [0, 3, 4],
#     [1, 6, 4]])
# print("normalizeRows(x) = " + str(normalizeRows(x)))


# GRADED FUNCTION: softmax
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    exp_x = np.exp(x)
    s = exp_x/np.sum(exp_x, axis=1, keepdims=True)

    return s


# x = np.array([
#     [9, 2, 5, 0, 0],
#     [7, 5, 0, 0 ,0]])
# print("softmax(x) = " + str(softmax(x)))


def test():
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    ### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot+= x1[i]*x2[i]
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
    tic = time.process_time()
    outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
    for i in range(len(x1)):
        for j in range(len(x2)):
            outer[i,j] = x1[i]*x2[j]
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### CLASSIC ELEMENTWISE IMPLEMENTATION ###
    tic = time.process_time()
    mul = np.zeros(len(x1))
    for i in range(len(x1)):
        mul[i] = x1[i]*x2[i]
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
    W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
    tic = time.process_time()
    gdot = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        for j in range(len(x1)):
            gdot[i] += W[i, j]*x1[j]
    toc = time.process_time()
    print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    ### VECTORIZED DOT PRODUCT OF VECTORS ###
    tic = time.process_time()
    dot = np.dot(x1,x2)
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED OUTER PRODUCT ###
    tic = time.process_time()
    outer = np.outer(x1,x2)
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED ELEMENTWISE MULTIPLICATION ###
    tic = time.process_time()
    mul = np.multiply(x1,x2)
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED GENERAL DOT PRODUCT ###
    tic = time.process_time()
    dot = np.dot(W,x1)
    toc = time.process_time()
    print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# GRADED FUNCTION: L1
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    loss = np.sum(np.abs(yhat-y))

    return loss


# yhat = np.array([.9, 0.2, 0.1, .4, .9])
# y = np.array([1, 0, 0, 1, 1])
# print("L1 = " + str(L1(yhat,y)))


# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """

    # loss = np.sum(np.dot(yhat-y, yhat-y))
    loss = np.sum(np.power(yhat - y, 2))

    return loss


# yhat = np.array([.9, 0.2, 0.1, .4, .9])
# y = np.array([1, 0, 0, 1, 1])
# print("L2 = " + str(L2(yhat,y)))
# print("power=="+str(np.power(yhat, 2)))




