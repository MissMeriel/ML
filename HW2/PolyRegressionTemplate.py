# Machine Learning HW2 Poly Regression
#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


test = True

# Step 1
# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    # your code
    x = np.array([])
    y = np.array([])
    line_count = 0
    features = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.split("\t")
            print(line)
            features = len(line)
            x = np.append(x, float(line[0]))
            y = np.append(y, float(line[-1].replace("\n", "")))
            line_count += 1
    f.close()
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    # your code
    x_t = np.transpose(x)
    xtx = np.dot(x_t, x)
    xtx = np.array(xtx)
    inv = []
    try:
        xtx.reshape(1,1)
        inv = [(1/xtx)]
    except ValueError:
        # print("not a 1x1 matrix")
        inv = np.linalg.inv(xtx)
    x_ty = np.dot(x_t, y)
    theta = np.dot(inv, x_ty)
    return theta

# Step 2: 
# Given a n by 1 dimensional array return an n by num_dimension array
# consisting of [1, x, x^2, ...] in each row
# x: input array with size n
# degree: degree number, an int
# result: polynomial basis based reformulation of x 
def increase_poly_order(x, degree):
    # your code
    result = np.ones(degree+1)
    i = 0
    for d in range(1, degree+1):
        result[d] = pow(x, d)
    return result

# split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
# Note: don't need to randomize the split bc data is randomized already
def train_test_split(x, y, train_proportion):
    index = int(len(x) * train_proportion)
    x_train = x[:index]
    x_test = x[index:]
    y_train = y[:index]
    y_test = y[index:]
    return x_train, x_test, y_train, y_test

# Find theta using the gradient descent/normal equation
def solve_regression(x, y, method):
    if(method=='G'):
        learning_rate = 0.1
        num_iterations = 1000
        # your GD code from HW1 or better version that returns best theta as well as theta at each epoch
        m = len(y)
        thetas = np.array(np.random.randn(2))
        for iters in range(num_iterations):
            gradients = (1.0 / m) * np.dot(np.transpose(x), np.dot(x, thetas) - y)
            thetas = thetas - learning_rate * gradients
        theta = thetas[-1]
    else:
        thetas = []
        theta = normal_equation(x, y)
    return theta, thetas 

# Given an array of y and y_predict return loss
# y: an array of size n
# y_predict: an array of size n
# loss: a single float
def get_loss(y, y_predict):
    # your code
    loss = 0
    n = len(y)
    for i in range(n):
        loss += (1.0 / n) * (y_predict[i] - y[i]) ** 2
    return loss

# Given an array of x and theta predict y
# x: an array with size n x d
# theta: np array including parameters
# y_predict: prediction labels, an array with size n
def predict(x, theta):
    # your code
    return np.dot(x, theta)

# Given a list of thetas one per (s)GD epoch
# this creates plots of epoch vs prediction loss (one about train, and another about validation or test)
# this figure checks GD optimization traits of the best theta
def plot_epoch_losses(x_train, x_test, y_train, y_test, best_thetas, title):
    losses = []
    tslosses = []
    epochs = []
    epoch_num = 1
    for theta in best_thetas:
        tslosses.append(get_loss(y_train, predict(x_train, theta)))
        losses.append(get_loss(y_test, predict(x_test, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses, label="training_loss")
    plt.plot(epochs, tslosses, label="testing_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()


# Given a list of degrees.
# For each degree in the list, train a polynomial regression.
# Return training loss and validation loss for a polynomial regression of order degree for
# each degree in degrees. 
# Use 60% training data and 20% validation data. Leave the last 20% for testing later.
# Input:
# x: an array with size n x d
# y: an array with size n
# degrees: A list of degrees
# Output:
# training_losses: a list of losses on the training dataset
# validation_losses: a list of losses on the validation dataset
def get_loss_per_poly_order(x, y, degrees):
    # your code
    n = x.shape[0]
    x_train, x_val, y_train, y_val = train_test_split(x, y, 0.75)
    training_losses = np.zeros(len(degrees))
    validation_losses = np.zeros(len(degrees))
    deg_count = 0
    for deg in degrees:
        # calculate training losses
        x_train_degreed = np.ones((len(x_train), deg+1))
        for i in range(0,len(x_train)):
            x_degreed = increase_poly_order(x_train[i], deg)
            x_train_degreed[i] = x_degreed
        # print("x_train_degreed.shape: "+str(x_train_degreed.shape))
        theta, thetas = solve_regression(x_train_degreed, y_train, 'N')
        y_predict = np.dot(x_train_degreed, theta)
        # print("y_train.shape: "+str(y_train.shape))
        # print("y_predict.shape: "+str(y_predict.shape))
        training_losses[deg_count] = get_loss(y_train, y_predict)

        # calculate validation losses
        x_val_degreed = np.ones((len(x_val), deg+1))
        for i in range(0,len(x_val)):
            x_degreed = increase_poly_order(x_val[i], deg)
            x_val_degreed[i] = x_degreed
        y_predict = np.dot(x_val_degreed, theta)
        validation_losses[deg_count] = get_loss(y_val, y_predict)
        deg_count += 1
    # print("training losses:"+str(training_losses))
    # print("validation_losses:" + str(validation_losses))
    return training_losses, validation_losses

# Give the parameter theta, best-fit degree , plot the polynomial curve
def best_fit_plot(theta, degree, x, y, x_orig, y_orig):
    # your code
    x_min = np.min(x_orig)
    x_max = np.max(x_orig)
    range_arr = np.linspace(x_min, x_max, 1000)
    x_best = np.zeros(len(range_arr))
    y_best = np.zeros(len(range_arr))
    i = 0
    for x_index in range_arr:
        x_best[i] = x_index
        #print("i:"+str(x_index)+", degree:"+str(degree))
        x_i_degreed = increase_poly_order(x_index, degree)
        #print("theta "+str(theta))
        y_best[i] = np.dot(x_i_degreed, theta)
        i += 1
    plt.scatter(x_orig, y_orig)
    #print("x_best: "+str(x_best))
    plt.plot(x_best, y_best, label="best fit plot; degree "+str(degree))
    plt.legend(loc='best')
    plt.title("best fit plot")
    plt.show()
    #print("")
    #print("best fit plot")


# Give the parameter theta, best-fit degree , plot the polynomial curve
def best_fit_plot2(theta, degree, x, y, x_orig, y_orig):
    # your code
    x_min = np.min(x_orig)
    x_max = np.max(x_orig)
    range_arr = np.linspace(x_min, x_max, 1000)
    x_best = np.zeros(len(range_arr))
    y_best = np.zeros(len(range_arr))
    i = 0
    for x_index in range_arr:
        x_best[i] = x_index
        #print("i:"+str(x_index)+", degree:"+str(degree))
        x_i_degreed = increase_poly_order(x_index, degree)
        #print("theta "+str(theta))
        y_best[i] = np.dot(x_i_degreed, theta)
        i += 1
    plt.scatter(x_orig, y_orig)
    #print("x_best: "+str(x_best))
    plt.plot(x_best, y_best, label="best fit plot; degree "+str(degree))
    plt.legend(loc='best')
    plt.title("best fit plot")
    plt.show()
    #print("")
    #print("best fit plot")

def select_hyperparameter(degrees, x_train, x_test, y_train, y_test):
    # Part 1: hyperparameter tuning:  
    # Given a set of training examples, split it into train-validation splits
    # do hyperparameter tune  
    # come up with best model, then report error for best model
    training_losses, validation_losses = get_loss_per_poly_order(x_train, y_train, degrees)
    plt.plot(degrees, training_losses, label="training_loss")
    plt.plot(degrees, validation_losses, label="validation_loss")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.title("poly order vs validation_loss")
    plt.show()

    # Part 2:  testing with the best learned theta 
    # Once the best hyperparameter has been chosen 
    # Train the model using that hyperparameter with all samples in the training 
    # Then use the test data to estimate how well this model generalizes.
    d = np.where(validation_losses == min(validation_losses))
    best_degree = degrees[int(d[0])] # fill in using best degree from part 2
    x_train_degreed = x_to_x_degreed(x_train, best_degree)
    best_theta, best_thetas = solve_regression(x_train_degreed, y_train, 'N')
    best_fit_plot(best_theta, best_degree, x_test, y_test, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)))
    #print(best_theta)
    x_test_degreed = x_to_x_degreed(x_test, best_degree)
    test_loss = get_loss(y_test, predict(x_test_degreed, best_theta))
    train_loss = get_loss(y_train, predict(x_train_degreed, best_theta))

    # Part 3: visual analysis to check GD optimization traits of the best theta 
    print(best_degree)
    x_train_p = np.ones((len(x_train), best_degree+1))
    for i in range(0, len(x_train)):
        x_train_p[i] = increase_poly_order(x_train[i], best_degree)
    x_test_p = np.ones((len(x_test), best_degree+1))
    for i in range(0, len(x_test)):
        x_test_p[i] = increase_poly_order(x_test[i], best_degree)
    gbest_thetas = gradient_descent(x_train_p, y_train, 0.005, 2000)
    best_fit_plot(gbest_thetas[-1], best_degree, x_train, y_train, x, y)
    plot_epoch_losses(x_train_p, x_test_p, y_train, y_test, gbest_thetas,
                      "best learned theta - train, test losses vs. GD epoch ")
    return best_degree, best_theta, train_loss, test_loss

def x_to_x_degreed(x, degree):
    x_degreed = np.ones((len(x), degree + 1))
    for i in range(0, len(x)):
        x_i_degreed = increase_poly_order(x[i], degree)
        x_degreed[i] = x_i_degreed
    return x_degreed

# Given a list of dataset sizes [d_1, d_2, d_3 .. d_k]
# Train a polynomial regression with first d_1, d_2, d_3, .. d_k samples
# Each time, 
# return the a list of training and testing losses if we had that number of examples.
# We are using 0.5 as the training proportion because it makes the testing_loss more stable
# in reality we would use more of the data for training.
# Input:
# x: an array with size n x d
# y: an array with size n
# example_num: A list of dataset size
# Output:
# training_losses: a list of losses on the training dataset
# testing_losses: a list of losses on the testing dataset
def get_loss_per_tr_num_examples(x, y, example_num, train_proportion):
    training_losses = []
    testing_losses = []
    for i in example_num:
        x_train, x_test, y_train, y_test = train_test_split(x[:i], y[:i], 0.5)
        theta = normal_equation(x_train, y_train)

        training_losses.append(get_loss(y_train, predict(x_train, theta)))
        testing_losses.append(get_loss(y_test, predict(x_test, theta)))
    return training_losses, testing_losses


# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    # initialize theta as [1 1]
    theta = np.zeros(np.size(x, 1))
    thetas = []
    for i in range(num_iterations):
        loss = np.dot(x, theta) - y
        gradient = np.dot(x.T,loss)
        gradient /= len(x) # normalize by number of examples
        theta = theta - learning_rate*gradient
        thetas.append(theta)
    return np.array(thetas)


if __name__ == "__main__":
    # select the best polynomial through train-validation-test formulation 
    x, y = load_data_set("dataPoly.txt")
    plt.scatter(x,y)
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    degrees = [i for i in range(1,10)]
    best_degree, best_theta, train_loss, test_loss = select_hyperparameter(degrees, x_train, x_test, y_train, y_test)
 
    # Part 4: analyze the effect of revising the size of train data: 
    # Show training error and testing error by varying the number for training samples 
    x, y = load_data_set("dataPoly.txt")
    x = x_to_x_degreed(x, 8)
    example_num = [10*i for i in range(2, 11)] # python list comprehension
    training_losses, testing_losses = get_loss_per_tr_num_examples(x, y, example_num, 0.5)
    plt.plot(example_num, training_losses, label="training_loss")
    plt.plot(example_num, testing_losses, label="testing_losses")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.title("number of examples vs training_loss and testing_loss")
    plt.show()
