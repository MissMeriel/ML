# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    # your code
    x = []
    y = np.array([])
    line_count = 0
    features = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.split("\t")
            temp = line[:len(line)-1]
            x.append([])
            x_subarr = [float(temp[i]) for i in range(len(temp))]
            x[line_count] = x_subarr
            y = np.append(y, float(line[-1].replace("\n", "")))
            line_count += 1
    f.close()
    return np.array(x), y

def bar_plot(best_beta):
    plt.bar([i for i in range(len(best_beta))], best_beta, align='center', alpha=0.5)
    plt.xlabel('beta indices')
    plt.ylabel('beta values')
    plt.title('bar plot of best beta')
    plt.show()

# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    # your code
    index = int(len(x) * train_proportion)
    x_train = x[:index]
    x_test = x[index:]
    y_train = y[:index]
    y_test = y[index:]
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
    # your code
    beta = 0
    inv = 0
    xt = np.transpose(x)
    x_sq = np.dot(xt, x)
    id_mat = np.identity(int(xt.shape[0]))
    try:
        x_sq.reshape(1, 1)
        inv = [(1 / (x_sq + lambdaV))]
    except ValueError:
        # print("not a 1x1 matrix")
        inv = np.linalg.pinv(x_sq + lambdaV * id_mat)
    beta = np.dot(np.dot(inv, xt), y)
    return beta

# Extra Credit: Find theta using gradient descent
def gradient_descent(x, y, lambdaV, learning_rate, num_iterations):
    # your code
    # initialize theta as [1 1]
    v = np.size(x, 1)
    theta = np.zeros(np.size(x, 1))
    thetas = []
    for i in range(num_iterations):
        loss = np.dot(x, theta) - y
        regularization_term = learning_rate * 2 * lambdaV * (theta **2)
        #regularization_term = 2 * lambdaV * theta
        gradient = np.dot(x.T, loss)
        gradient /= len(x)  # normalize by number of examples
        gradient -= regularization_term
        #gradient += regularization_term
        theta = theta - learning_rate * gradient
        thetas.append(theta)
        beta = theta
    return beta

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    loss = 0
    n = len(y)
    for i in range(n):
        loss +=  (1.0 / n) * (y_predict[i] - y[i]) ** 2
        # loss += np.linalg.norm(y_predict[i] - y[i])
    return loss

# Given an array of x and theta predict y
def predict(x, theta):
    # your code
    return np.dot(x, theta)

def l2norm(theta):
    return (sum(theta ** 2))** 0.5

# Find the best lambda given x_train and y_train using 4 fold cv
# Note: k = number of bins, iteratively pick one bin as your test set
def cross_validation(x_train, y_train, lambdas):
    valid_losses = np.zeros(len(lambdas))
    training_losses = np.zeros(len(lambdas))
    # your code
    # split into 4
    split_x = np.array(np.split(x_train, 4))
    split_y = np.array(np.split(y_train, 4))
    lambda_count = 0
    for l in lambdas:
        loocv_tr_loss = np.ones(4)
        loocv_val_loss = np.ones(4)
        for i in range(4):
            # train on one of the four bins
            x_val = split_x[i]
            y_val = split_y[i]
            x_tr = np.concatenate(np.delete(np.copy(split_x), i, 0))
            y_tr = np.concatenate(np.delete(np.copy(split_y), i, 0))
            beta = normal_equation(x_tr, y_tr, l)
            # get training loss
            y_predict = predict(x_tr, beta)
            #loocv_tr_loss += get_loss(y_tr, y_predict) / 4.0
            loocv_tr_loss[i] = get_loss(y_tr, y_predict)
            # get validation loss
            y_predict = predict(x_val, beta)
            #loocv_val_loss += get_loss(y_val, y_predict) / 4.0
            loocv_val_loss[i] = get_loss(y_val, y_predict)
        training_losses[lambda_count] = np.mean(loocv_tr_loss)
        valid_losses[lambda_count] = np.mean(loocv_val_loss)
        lambda_count += 1
    return np.array(valid_losses), np.array(training_losses)


def step3(x_train, y_train, lambdas):
    training_losses = np.zeros(len(lambdas))
    # your code
    lambda_count = 0
    for l in lambdas:
        beta = normal_equation(x_train, y_train, l)
        # get training loss
        y_predict = predict(x_train, beta)
        training_losses[lambda_count] = get_loss(y_train, y_predict)
        lambda_count += 1
    return np.array(training_losses)


if __name__ == "__main__":
    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 20)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=2**10)
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]

    # step 2: analysis 
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = l2norm(normal_beta) # your code get l2 norm of normal_beta
    best_beta_norm = l2norm(best_beta) # your code get l2 norm of best_beta
    large_lambda_norm = l2norm(large_lambda_beta) # your code get l2 norm of large_lambda_beta
    print("best lambda: "+str(best_lambda))
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    bar_plot(best_beta)


    # Step3: Retrain a new model using all sampling in training, then report error on testing set
    # your code ! 
    training_losses_no_val = step3(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses_no_val[1:], label="training_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training loss without validation")
    plt.show()

    # plot to compare with validation versus without
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.plot(lambdas[1:], training_losses_no_val[1:], label="training_loss_no_valid")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss and training loss without validation")
    plt.show()

    # Step Extra Credit: Implement gradient descent, analyze and show it gives the same or very similar beta to normal_equation
    # to prove that it works
    best_gradient_beta = gradient_descent(x_train, y_train, best_lambda, 0.005, 2000)
    print(best_gradient_beta[0:5])
    print(best_beta[0:5])
    # print("best_gradient_beta.shape: "+str(best_gradient_beta.shape))
    # print(best_gradient_beta - best_beta)
    gradient_loss = get_loss(y_test, predict(x_test, best_gradient_beta))
    print("gradient_loss: "+str(gradient_loss))