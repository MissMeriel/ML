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

#def bar_plot():


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
    try:
        x_sq.reshape(1, 1)
        inv = [(1 / (x_sq + lambdaV))]
    except ValueError:
        # print("not a 1x1 matrix")
        inv = np.linalg.inv(x_sq + lambdaV * np.identity(int(xt.shape[0])))
    beta = np.dot(np.dot(inv, xt), y)
    return beta

# Extra Credit: Find theta using gradient descent
def gradient_descent(x, y, lambdaV, num_iterations, learning_rate):
    # your code

    return beta

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    loss = 0
    n = len(y)
    for i in range(n):
        temp = (1.0 / n) * (y_predict[i] - y[i]) ** 2
        loss += temp
    return loss

# Given an array of x and theta predict y
def predict(x, theta):
    # your code
    y_predict = np.dot(x, theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv
# Note: k = number of bins, iteratively pick one bin as your test set
def cross_validation(x_train, y_train, lambdas):
    print("lambdas: "+str(lambdas))
    valid_losses = np.zeros(len(lambdas))
    training_losses = np.zeros(len(lambdas))
    # your code
    # split into 4
    split_x = np.array(np.split(x_train, 4))
    split_y = np.array(np.split(y_train, 4))
    print("split_x.shape:" + str(split_x.shape))
    print("split_y.shape" + str(split_y.shape))
    print("x_train.shape:" + str(x_train.shape))
    print("y_train.shape" + str(y_train.shape))
    lambda_count = 0
    for l in lambdas:
        loocv_tr_loss = 0
        loocv_val_loss = 0
        for i in range(4):
            # train on one of the four bins
            print("train on one of the four bins")
            x_val = split_x[i]
            y_val = split_y[i]
            x_tr = np.concatenate(np.delete(np.copy(split_x), i, 0))
            y_tr = np.concatenate(np.delete(np.copy(split_y), i, 0))
            print("start normal eqn")
            print("x_tr.shape:"+str(x_tr.shape))
            print("y_tr.shape" + str(y_tr.shape))
            #print("" + str())
            beta = normal_equation(x_tr, y_tr, l)
            print("done normal eqn")
            # get training loss
            y_predict = predict(x_tr, beta)
            loocv_tr_loss += get_loss(y_tr, y_predict) / 4.0
            # get validation loss
            y_predict = predict(x_val, beta)
            loocv_val_loss += get_loss(y_val, y_predict) / 4.0
        training_losses[lambda_count] = loocv_tr_loss
        valid_losses[lambda_count] = loocv_val_loss
        lambda_count += 1
        print("lambda_count: "+str(lambda_count))
    return np.array(valid_losses), np.array(training_losses)

if __name__ == "__main__":  
    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    print(type(x[1][1]))
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
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
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]

    # step 2: analysis 
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = np.linalg.norm(normal_beta, 2) # your code get l2 norm of normal_beta
    best_beta_norm = np.linalg.norm(best_beta, 2) # your code get l2 norm of best_beta
    large_lambda_norm = np.linalg.norm(large_lambda_beta, 2) # your code get l2 norm of large_lambda_beta
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


    # Step Extra Credit: Implement gradient descent, analyze and show it gives the same or very similar beta to normal_equation
    # to prove that it works
