import keras as keras
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

MODEL_EXPORT_PATH = 'trained_model/'
FIG_NAME = 'resultado_treinamento'

# passo 0
def le_arquivo(nome_arq):
    arq = open(nome_arq, 'r')  # abre o arquivo
    csvreader = csv.reader(arq, delimiter=",")
    headers = next(csvreader, None)
    rows = []
    for row in csvreader:
        rows.append(row)

    arq.close()

    rows = np.array(rows)

    rows = rows.astype(float)

    return rows


ct = le_arquivo('resumo.csv')

num_padroes, num_atrib = ct.shape

X = ct[:, 0:num_atrib - 1]
Y = ct[:, num_atrib - 1:num_atrib]

XTraining, XValidation, YTraining, YValidation = train_test_split(X,Y,stratify=Y,test_size=0.1) # before model building

n0 = 200
n1 = 200

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=[num_atrib - 1]))
model.add(keras.layers.Dense(n0, activation="relu"))
model.add(keras.layers.Dense(n1, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

ts = 20
lr = 0.1

optimizer = keras.optimizers.Adam(learning_rate=lr, decay=lr / ts)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(XTraining, YTraining, epochs=ts, validation_data=(XValidation,YValidation), shuffle=True)

model.save(MODEL_EXPORT_PATH)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2)
plt.savefig(FIG_NAME)

x_new = X[1:2, :]

print(x_new)

y_new = model.predict(x_new)

print(y_new)