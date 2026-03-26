import tensorflow as tf

import numpy as np
import pandas as pd

import random

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, Dropout
from keras.optimizers import Adam

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


SEED = 39
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


class CleanPrintCallback(Callback):
    def __init__(self, every_n_epochs=10):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # v2: ređi ispis da trening ne zatrpava i deluje "zalepljeno"
        if (epoch + 1) % self.every_n_epochs != 0 and epoch != 0:
            return
        line = f"Epoch {epoch+1}/{self.params['epochs']} - "
        line += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        print(line)


def to_categorical(y, num_classes):
    y = np.array(y, dtype=int)
    one_hot = np.zeros((y.size, num_classes), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot


df = pd.read_csv(
    r"/Users/4c/Desktop/GHQ/data/loto7_4586_k24.csv",
    sep=',',
    quoting=1,
    skip_blank_lines=True,
    header=None
)


sequences = []

for index, row in df.iterrows():
    seq = []
    for value in row:
        try:
            num = int(float(value))  # radi i sa "1" i 1.0
            seq.append(num)
        except:
            pass
    if len(seq) == 7:
        sequences.append(seq)


sequences = np.array(sequences)

print()
print("broj kombinacija shape:", sequences.shape)
print()
"""
broj kombinacija shape: (4554, 7)
"""

if sequences.shape[1] != 7:
    raise RuntimeError("svaka kombinacija mora da ima 7 brojeva")

num_classes = 39  # Srbija Loto brojevi 1–39


window_size = 7

X, y_raw = [], []

for i in range(len(sequences) - window_size):
    X.append(sequences[i:i+window_size])
    y_raw.append(sequences[i+window_size]) 

X = np.array(X) 
y_raw = np.array(y_raw) 
print()
print("X shape:", X.shape)
print("y_raw shape:", y_raw.shape)
print()
"""
X shape: (4549, 5, 7)
y_raw shape: (4549, 7)
"""


y_onehot = np.array([to_categorical(seq-1, num_classes=num_classes) for seq in y_raw])


input_layer = Input(shape=(window_size, 7))
x = Masking(mask_value=0)(input_layer)
x = LSTM(128, activation='tanh', return_sequences=False)(x)
x = Dropout(0.2)(x)


outputs = [Dense(num_classes, activation='softmax', name=f'num_{i+1}')(x) for i in range(7)]


model = Model(inputs=input_layer, outputs=outputs)


model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='categorical_crossentropy'
)


print()
model.summary()
print()
"""
Model: "functional"
...
"""


y_list = [y_onehot[:, i, :] for i in range(7)]  
# Keras zahteva listu izlaza


print()
print("Broj uzoraka:", X.shape[0])
print()
"""
Broj uzoraka: 4549
"""

callbacks = [
    CleanPrintCallback(every_n_epochs=10),
    EarlyStopping(monitor="loss", patience=40, restore_best_weights=True, min_delta=1e-5),
    ReduceLROnPlateau(monitor="loss", factor=0.5, patience=12, min_lr=1e-6, verbose=1),
]

print()
model.fit(
    X.astype('float32'),
    [y_onehot[:, i, :].astype('float32') for i in range(7)],
    epochs=600,
    batch_size=32,
    verbose=0,  # iskljuci standardni verbose
    callbacks=callbacks
)

print()
"""
Epoch 1/100 - loss: 24.8857 - num_1_loss: 3.3484 - num_2_loss: 3.5510 - num_3_loss: 3.5787 - num_4_loss: 3.6725 - num_5_loss: 3.6697 - num_6_loss: 3.6065 - num_7_loss: 3.4474
...
Epoch 100/100 - loss: 20.2468 - num_1_loss: 2.4999 - num_2_loss: 2.9494 - num_3_loss: 3.1233 - num_4_loss: 3.1659 - num_5_loss: 3.1073 - num_6_loss: 2.9317 - num_7_loss: 2.4557

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step
"""


def predict_next_sequence(model, last_window):

    last_window = np.array([last_window]) 
    preds = model.predict(last_window, verbose=0)

    predicted_numbers = []
    for p in preds:
        number = int(np.argmax(p[0]) + 1)
        predicted_numbers.append(number)

    # v2: zadrži rastući redosled po pozicijama 7/39
    predicted_numbers = sorted(predicted_numbers)
    for i in range(len(predicted_numbers)):
        lo = i + 1
        hi = 33 + i
        predicted_numbers[i] = int(np.clip(predicted_numbers[i], lo, hi))

    return predicted_numbers


last_window = sequences[-window_size:]
predicted_sequence = predict_next_sequence(model, last_window)
print()
print("\nPREDICT SRBIJA LOTO 7/39:", predicted_sequence)
print()
"""
100 Epoch
PREDICT SRBIJA LOTO 7/39: [1, 11, x, y, z, 34, 38]

4554 Epoch
PREDICT SRBIJA LOTO 7/39: [13, 2, x, y, z, 23, 17]
"""

"""
V2 poboljšanja:

window_size: 5 -> 7
LSTM jači: 100 -> 128 + Dropout(0.2)
stabilniji trening: Adam(3e-4), EarlyStopping, ReduceLROnPlateau
manje spam-a u logu: CleanPrintCallback(every_n_epochs=10)
predikcija tiša (model.predict(..., verbose=0))
izlaz sređen kao rastuća 7/39 kombinacija (sort + clip po poziciji)
"""
