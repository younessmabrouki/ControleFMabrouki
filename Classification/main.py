import os
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np



def analyse_data(path):
	total = 0
	for subpath, directories, files in os.walk(path):
		if len(files):
			print("classe : {}, nb exemples : {}".format(subpath, len(files)))
			total += len(files)
		else:
			print("nb classes : {}".format(len(directories)))
	print("nombre d'exemples total : {}".format(total))

def create_model_light(input_shape):
	model = Sequential()
	model.add(Conv2D(128, (3,3), activation="relu", input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dense(64, activation="relu"))
	model.add(Dense(32, activation="relu"))
	model.add(Dense(16, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])
	return model

def create_model_full(input_shape):
	model = Sequential()
	model.add(Conv2D(128, (3,3), activation="relu", input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dense(64, activation="relu"))
	model.add(Dropout(0.8))
	model.add(Dense(47, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])
	return model

def load_generators(path, batch_size, ratio):
	data_generator = ImageDataGenerator(rescale=1./255., validation_split=ratio)
	train_generator = data_generator.flow_from_directory(
		directory = path,
		target_size = (150, 150), shuffle=False, class_mode="categorical",
		batch_size = batch_size, subset="training"
	)
	validation_generator = data_generator.flow_from_directory(
		directory = path,
		target_size = (150, 150), shuffle=False, class_mode="categorical",
		batch_size = batch_size, subset="validation"
	)
	return (train_generator, validation_generator)

def main():
	path = "data/dataClassif"
	analyse_data(path)
	validation_percent = 0.33
	samples = 20934
	nb_validation_samples = samples * validation_percent
	nb_train_samples = samples - nb_validation_samples
	epochs = 10
	batch_size = 32
	input_shape = (150, 150, 3)
	filename = "results"
	modelname = "simpsons_model_small"
	train_generator, validation_generator = load_generators(path, batch_size, validation_percent)
	model = create_model_full(input_shape)
	history = model.fit(train_generator, steps_per_epoch = nb_train_samples // batch_size, epochs = epochs, validation_data = validation_generator, validation_steps = nb_validation_samples // batch_size)
	print(history.history.keys())
	plt.plot(history.history["accuracy"])
	plt.plot(history.history["val_accuracy"])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(["train", "test"], loc="upper left")
	plt.savefig(str("%s_accuracy.png" % filename))
	print("training")
	preds = model.predict(train_generator, len(train_generator))
	y_pred = np.argmax(preds, axis=1)
	print(confusion_matrix(train_generator.classes, y_pred))
	model.save(modelname)
	"""
	step per epoch c'est des petits groupes d'images, petit groupe par petit groupe,
	évite de péter la ram
	nombre total d'images bdd / taille du sous groupe
	"""
main()