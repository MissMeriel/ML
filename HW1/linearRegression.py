# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    x = np.array([])
    y = np.array([])
    with open(filename, "r") as f:
        for line in f:
            line = line.split("\t")
            x = np.append(x, np.array([float(line[0]), float(line[1])]))
            y = np.append(y, float(line[2].replace("\n", "")))
    f.close()
    x = x.reshape((200,2))
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    x_t = np.transpose(x)
    inv = np.linalg.inv(np.dot(x_t, x))
    x_ty = np.dot(x_t, y)
    theta = np.dot(inv, x_ty)
    return theta

# Find thetas using stochiastic gradient descent
# Don't forget to shuffle
def stochiastic_gradient_descent(x, y, learning_rate, num_iterations):
    m = len(y)
    thetas = np.random.randn(2)
    for iters in range(num_iterations):
        #shuffle x's and y's
        indices = [i for i in range(m)]
        indices = np.array(indices)
        print("indices:"+str(indices))
        indices = np.random.shuffle(indices)
        print("shuffled indices:"+str(indices))
        for i in range(m):
            gradients = (1.0/m)*np.dot(np.transpose(x[i]), np.dot(x[i], thetas)-y[i])
            thetas = thetas - learning_rate * gradients
    return thetas

# see p118 ML book
#Hands on Machine learning with scikit-learn and tensorflow
# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    #print("x.shape:"+str(x.shape))
    #print("y.shape:"+str(y.shape))
    m = len(y)
    thetas = np.array(np.random.randn(2))
    #print("thetas:"+str(thetas))
    for iters in range(num_iterations):
        gradients = (1.0/m)*np.dot(np.transpose(x), np.dot(x, thetas)-y)
        thetas = thetas - learning_rate * gradients
    return thetas

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_iterations, batch_size):
    # your code
    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   y_predict = np.dot(x, theta)
   return y_predict

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    loss = 0
    n = len(y)
    for i in range(n):
        loss += (1.0/n) * (y_predict[i] - y[i])**2
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    x, y = load_data_set('regression-data.txt')
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    print("Normal Eqn Theta:"+str(theta))
    plot(x, y, theta, "Normal Equation Best Fit")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = gradient_descent(x, y, 0.5, 1000)
    print("Gradient Descent Theta:"+str(thetas))
    print("thetas[-1]:"+str(thetas[-1]))
    plot(x, y, thetas, "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = stochiastic_gradient_descent(x, y, 0, 0) # Try different learning rates and number of iterations
    plot(x, y, thetas, "Stochiastic Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Stochiastic Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = minibatch_gradient_descent(x, y, 0, 0, 0)
    plot(x, y, thetas, "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Mean Epoch vs Training Loss")