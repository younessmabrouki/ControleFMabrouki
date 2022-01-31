import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def load_data(file):
  return pd.read_csv(file)

def analyze_data(data):
  print(data.shape)
  print(data.info())
  print(data.describe())
  print(data.head(5))
  print(data.value_counts)

def splitDataToXAndY(data):
  X = data.iloc[:, 0:(len(data.columns)-1)].values
  Y = data.iloc[:, (len(data.columns)-1)].values
  return X, Y

def splitDataToTrainAndTest(X, Y):
  return train_test_split(X, Y, test_size = 0.333, random_state = 1337)

def encodeData(X, Y):
  labelencoder = LabelEncoder()
  Y = labelencoder.fit_transform(Y)
  return X, Y

def create_model(input_shape):
  model = Sequential()
  model.add(Dense(10, activation='relu', input_shape=(input_shape,)))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=500, toSave=True):
  if toSave :
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    model.save("nn_backup")
  else:
    model = keras.models.load_model("nn_backup")
  return model

def create_tfidf(x_train, x_test):
  tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.01, max_df=1.0, stop_words='english', ngram_range=(1,1))
  tfidf = tfidf.fit(x_train)
  x_train_tfidf = tfidf.transform(x_train).toarray()
  x_test_tfidf = tfidf.transform(x_test).toarray()
  return x_train_tfidf, x_test_tfidf, tfidf

def main():
  data = load_data("data/text.csv")
  X, Y = splitDataToXAndY(data)
  X, Y = encodeData(X, Y)
  x_train, x_test, y_train, y_test = splitDataToTrainAndTest(X, Y)
  x_trainV, x_testV, vectorizer = create_tfidf(x_train.flatten(), x_test.flatten())
  model = create_model(x_trainV.shape[1])
  model = train_model(model, x_trainV, y_train, x_testV, y_test, 100, 500, False)
  y_pred = np.round(model.predict(x_testV))
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
main()