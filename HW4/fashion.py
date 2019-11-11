# Machine Learning Homework 4 - Image Classification

__author__ = '**'

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier


### Already implemented
def get_data(datafile):
	dataframe = pd.read_csv(datafile)
	dataframe = shuffle(dataframe)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	images /= 255
	return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
	layer1 = trained_model.layers[0]
	weights = layer1.get_weights()[0]

	# Feel free to change the color scheme
	colors = 'hot' if hot else 'binary'
	try:
		os.mkdir('weight_visualizations')
	except OSError:
		pass
	for i in range(num_to_display):
		wi = weights[:,i].reshape(28, 28)
		plt.imshow(wi, cmap=colors, interpolation='nearest')
		if save:
			plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
		else:
			plt.show()


### Already implemented
def output_predictions(predictions):
	with open('predictions.txt', 'w+') as f:
		for pred in predictions:
			f.write(str(pred) + '\n')


def plot_history(history):
	train_loss_history = history.history['loss']
	val_loss_history = history.history['val_loss']

	train_acc_history = history.history['accuracy']
	val_acc_history = history.history['val_accuracy']

	# plot


def create_mlp(args=None):
	# You can use args to pass parameter values to this method

	# Define model architecture
	model = Sequential()
	model.add(Dense(units=10, activation=args['activation'], input_dim=28 * 28))
	# add more layers...
	for i in range(3):
		model.add(Dense(units=10, activation=args['activation']))

	# Define Optimizer
	optimizer = keras.optimizers.SGD(lr=args['learning_rate'], decay=1e-6, momentum=args['momentum'], nesterov=False)

	# Compile
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model


def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	# num_classes?
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	model = create_mlp(args)

	early_stop = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
												baseline=None, restore_best_weights=False)]
	history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=early_stop)
	return model, history

def create_cnn(args=None):
	# You can use args to pass parameter values to this method

	# 28x28 images with 1 color channel
	input_shape = (28, 28, 1)

	# Define model architecture
	model = Sequential()
	model.add(Conv2D(filters=4, activation=args['activation'], kernel_size=args['kernel'], strides=(1, 1), input_shape=input_shape))
	# model.add(Conv2D(filters=112, activation=args['activation'], kernel_size=args['kernel'], strides=(1, 1), input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Flatten())
	for i in range(3):
		model.add(Dense(units=20, activation=args['activation']))
	model.add(Dense(units=10, activation=args['activation']))
	# Optimizer
	optimizer = keras.optimizers.SGD(lr=args['learning_rate'], decay=1e-6, momentum=args['momentum'], nesterov=args['nesterov'])

	# Compile
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	x_train = x_train.reshape(-1, 28, 28, 1)
	y_train = keras.utils.to_categorical(y_train) # , num_classes=10)
	model = create_cnn(args)
	# early_stop = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=True)]
	history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=None)
	return model, history


def train_and_select_model(train_csv, mlp=True):
	"""Optional method. You can write code here to perform a 
	parameter search, cross-validation, etc. """
	x_train, y_train = get_data(train_csv)

	best_model = None
	history = None
	best_validation_accuracy = 0

	if(mlp):
		learning_rates = [0.02, 0.03, 0.033, 0.04, 0.3, 1, 10]
		activations = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'relu']
		momentums = [0.9, 0.8, 0.85, 0.6, 0.5, 0.4]
		for i in range(4):
			args = {'learning_rate': learning_rates[i], 'activation': 'sigmoid', 'momentum': momentums[i]}
			model, history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
			validation_accuracy = history.history['val_accuracy'][-1]
			if (validation_accuracy > best_validation_accuracy):
				best_model = model
				best_validation_accuracy = validation_accuracy

	else:
		cnn_learning_rates = [0.01, 0.001, 1]
		activations = ['softplus', 'softplus', 'softplus']
		momentums = [0.5, 0.6, 0.7]
		nesterov = [False, False, False]
		kernels = [(5, 5), (6, 6), (7, 7)]
		strides = []
		pool_sizes = []
		filters = []
		for i in range(3):
			args = {'learning_rate': cnn_learning_rates[i], 'activation': activations[i], 'momentum': momentums[i],
					'nesterov': nesterov[i], 'kernel': kernels[0]}
			model, history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)
			validation_accuracy = history.history['val_accuracy'][-1]
			if (validation_accuracy > best_validation_accuracy):
				best_model = model
				best_validation_accuracy = validation_accuracy
	return best_model, history

def train_best_model(x_train, y_train):
	input_shape = (28, 28, 1)
	# Define model architecture
	model = Sequential()
	model.add(Conv2D(filters=4, activation='softplus', kernel_size=(7, 7), strides=(1, 1), input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Flatten())
	for i in range(3):
		model.add(Dense(units=20, activation='softplus'))
	model.add(Dense(units=10, activation='softplus'))
	# Optimizer
	optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=False)

	# Compile
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	x_train = x_train.reshape(-1, 28, 28, 1)
	y_train = keras.utils.to_categorical(y_train)  # , num_classes=10)
	# model = create_cnn(args)
	# early_stop = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=True)]
	history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=None)
	return model, history

if __name__ == '__main__':
	### Before you submit, switch this to grading_mode = True and rerun ###
	grading_mode = True
	if grading_mode:
		# When we grade, we'll provide the file names as command-line arguments
		if (len(sys.argv) != 3):
			print("Usage:\n\tpython3 fashion.py train_file test_file")
			exit()
		train_file, test_file = sys.argv[1], sys.argv[2]

		x_train, y_train = get_data(train_file)
		x_test, y_test = get_data(test_file)

		# train your best model
		best_model, history = train_best_model(x_train, y_train)
		
		# use your best model to generate predictions for the test_file
		x_test = x_test.reshape(-1, 28, 28, 1)
		predictions = best_model.predict(x_test)
		output_predictions(predictions)

		# Include all of the required figures in your report. Don't generate them here.

	else:
		train_file, test_file = sys.argv[1], sys.argv[2]
		# MLP
		mlp_model, mlp_history = train_and_select_model(train_file)
		plot_history(mlp_history)
		visualize_weights(mlp_model, num_to_display=10, save=False)

		# CNN
		cnn_model, cnn_history = train_and_select_model(train_file, mlp=False)
		plot_history(cnn_history)