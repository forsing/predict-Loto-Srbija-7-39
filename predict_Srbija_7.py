import tensorflow as tf

import numpy as np
import pandas as pd

import random

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking
from keras.optimizers import Adam

from keras.callbacks import Callback


SEED = 39
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)



class CleanPrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        line = f"Epoch {epoch+1}/{self.params['epochs']} - "
        line += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        print(line)


def to_categorical(y, num_classes):
    y = np.array(y, dtype=int)
    one_hot = np.zeros((y.size, num_classes), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot


df = pd.read_csv(
    r"/Users/milan/Desktop/GHQ/data/loto7_4554_k8.csv",
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


window_size = 5

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
x = LSTM(100, activation='tanh', return_sequences=False)(x)


outputs = [Dense(num_classes, activation='softmax', name=f'num_{i+1}')(x) for i in range(7)]


model = Model(inputs=input_layer, outputs=outputs)


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy'
)


print()
model.summary()
print()
"""
Model: "functional"
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃               ┃              ┃  Param ┃              ┃
┃ Layer (type)  ┃ Output Shape ┃      # ┃ Connected to ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ input_layer   │ (None, 5, 7) │      0 │ -            │
│ (InputLayer)  │              │        │              │
├───────────────┼──────────────┼────────┼──────────────┤
│ not_equal     │ (None, 5, 7) │      0 │ input_layer… │
│ (NotEqual)    │              │        │              │
├───────────────┼──────────────┼────────┼──────────────┤
│ masking       │ (None, 5, 7) │      0 │ input_layer… │
│ (Masking)     │              │        │              │
├───────────────┼──────────────┼────────┼──────────────┤
│ any (Any)     │ (None, 5)    │      0 │ not_equal[0… │
├───────────────┼──────────────┼────────┼──────────────┤
│ lstm (LSTM)   │ (None, 100)  │ 43,200 │ masking[0][… │
│               │              │        │ any[0][0]    │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_1 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_2 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_3 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_4 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_5 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_6 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
├───────────────┼──────────────┼────────┼──────────────┤
│ num_7 (Dense) │ (None, 39)   │  3,939 │ lstm[0][0]   │
└───────────────┴──────────────┴────────┴──────────────┘
 Total params: 70,773 (276.46 KB)
 Trainable params: 70,773 (276.46 KB)
 Non-trainable params: 0 (0.00 B)
"""


y_list = [y_onehot[:, i, :] for i in range(7)]  
# Keras zahteva listu izlaza


print()
print("Broj uzoraka:", X.shape[0])
print()
"""
Broj uzoraka: 4549
"""


print()
model.fit(
    X.astype('float32'),
    [y_onehot[:, i, :].astype('float32') for i in range(7)],
    epochs=4554,
    batch_size=32,
    verbose=0,  # iskljuci standardni verbose
    callbacks=[CleanPrintCallback()]
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
    preds = model.predict(last_window)

    predicted_numbers = []
    for p in preds:
        number = int(np.argmax(p[0]) + 1)
        predicted_numbers.append(number)

    return predicted_numbers


last_window = sequences[-window_size:]
predicted_sequence = predict_next_sequence(model, last_window)
print()
print("\nPREDICT SRBIJA LOTO 7/39:", predicted_sequence)
print()
"""
100 Epoch
PREDICT SRBIJA LOTO 7/39: [1, 11, 15, 17, 21, 34, 38]

4554 Epoch
PREDICT SRBIJA LOTO 7/39: [13, 2, 6, 13, 21, 23, 17]
"""
