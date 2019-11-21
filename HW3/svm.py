# Starting code for UVA CS 4501 ML- SVM

import numpy as np
np.random.seed(37)
import random
import csv
from sklearn.svm import SVC
# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']
#<=50k == 0; >50k == 1
y_dict = {' <=50K':0, ' >50K':1}

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']
#x_dict = dict({col_names_x[i]: {}} for i in range(len(col_names_x)))
x_dict = dict.fromkeys(col_names_x)

# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables.
def load_data(csv_file_path):
    # your code here
    file = open(csv_file_path)
    csv_reader = csv.reader(file, delimiter=',')
    x = []
    y = []
    for row in csv_reader:
        x.append(row[0:len(row)-1])
        y.append(row[-1])
    for key in x_dict.keys():
        x_dict[key] = {}
    x = preprocess_x(np.array(x))
    y = preprocess_y(y)
    return np.array(x), np.array(y)

def preprocess_x(x):
    id_count_x = np.zeros(len(col_names_x))
    for i in range(x.shape[0]):
        if(i < 5):
            print(x[i])
        for j in range(x.shape[1]):
            try:
                if (i < 5):
                    print(col_names_x[j]+":"+str(x[i][j]))
                    print(x_dict[col_names_x[j]])
                    print(x_dict)
                x[i][j] = x_dict[col_names_x[j]][x[i][j]]
            except KeyError:
                id_count_x[j] += 1
                x_dict[col_names_x[j]][x[i][j]] = id_count_x[j]
                x[i][j] = id_count_x[j]
            except TypeError: # catch NoneType dict
                id_count_x[j] += 1
                x_dict[col_names_x[j]] = {x[i][j]: id_count_x[j]}
                x[i][j] = id_count_x[j]
    return x

def preprocess_y(y):
    for i in range(len(y)):
        y[i] = y_dict[y[i]]
    return y

def fold(x, y, i, nfolds):
    # your code
    split_x = np.array_split(x, nfolds)
    split_y = np.array(np.array_split(y, nfolds))
    x_test = split_x[i]
    y_test = split_y[i]
    x_train = np.concatenate(np.delete(np.copy(split_x), i, 0))
    y_train = np.concatenate(np.delete(np.copy(split_y), i, 0))
    return x_train, y_train, x_test, y_test

def calc_accuracy(y_predict, y):
    # your code
    acc = 0
    for i in range(len(y_predict)):
        diff = y[i] - y_predict[i]
        acc += 1 - abs(diff)
    # print("acc: "+str(acc))
    return acc / float(len(y_predict))

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x, y = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    # TODO: Customize param set
    param_set = [
                 {'kernel': 'rbf', 'C': 1, 'degree': 1},
                 {'kernel': 'rbf', 'C': 10, 'degree': 3},
                 {'kernel': 'rbf', 'C': 100, 'degree': 5},
                 {'kernel': 'rbf', 'C': 1000, 'degree': 7},
    ]
    #for c in range(2,5):
    for kern in ['linear', 'poly', 'rbf', 'sigmoid']:
        param_set.append({'kernel':kern, 'C':1, 'degree':3})
    # your code here
    # iterate over all hyperparameter configurations
    # TODO: figure out SVC for best_model and best_score
    # TODO: 3-fold cross validation
    best_score = 0
    best_model = 0
    for param in param_set:
        nfolds = 3
        model = SVC(C=param['C'], kernel=param['kernel'], degree=param['degree'])
        # perform 3 FOLD cross validation
        new_score = 0
        for i in range(nfolds):
            x_train, y_train, x_test, y_test = fold(x, y, i, nfolds)
            model.fit(x_train, y_train)
            y_predict = model.decision_function(x_test)
            # acc = calc_accuracy(y_predict, y_test)
            # check against package accuracy scoring
            mscore = model.score(x_test, y_test)
            new_score += mscore / float(nfolds)
            # print("method score: "+str(acc))
            # print("model score: "+str(mscore))
        if new_score > best_score:
            best_score = new_score
            best_model = model
        # print cv scores for every hyperparameter and include in pdf report
        print(param, new_score)
    # select best hyperparameter from cv scores, retrain model 
    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format 
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to 
    # return a trained model with best hyperparameter from 3-FOLD 
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter. 
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print("The best model was scored %.2f" % cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.