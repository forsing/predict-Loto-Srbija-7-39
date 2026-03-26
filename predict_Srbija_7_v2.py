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
    r"/data/loto7_4586_k24.csv",
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
broj kombinacija shape: (4586, 7)
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
X shape: (4579, 7, 7)
y_raw shape: (4579, 7)
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
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 7, 7)]               0         []                            
                                                                                                  
 masking (Masking)           (None, 7, 7)                 0         ['input_1[0][0]']             
                                                                                                  
 lstm (LSTM)                 (None, 128)                  69632     ['masking[0][0]']             
                                                                                                  
 dropout (Dropout)           (None, 128)                  0         ['lstm[0][0]']                
                                                                                                  
 num_1 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
 num_2 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
 num_3 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
 num_4 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
 num_5 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
 num_6 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
 num_7 (Dense)               (None, 39)                   5031      ['dropout[0][0]']             
                                                                                                  
==================================================================================================
Total params: 104849 (409.57 KB)
Trainable params: 104849 (409.57 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
"""


y_list = [y_onehot[:, i, :] for i in range(7)]  
# Keras zahteva listu izlaza


print()
print("Broj uzoraka:", X.shape[0])
print()
"""
Broj uzoraka: 4579
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
...
Epoch 580/600 - loss: 8.8665 - num_1_loss: 1.3235 - num_2_loss: 1.2667 - num_3_loss: 1.2342 - num_4_loss: 1.2157 - num_5_loss: 1.2006 - num_6_loss: 1.3195 - num_7_loss: 1.3063
Epoch 590/600 - loss: 8.7746 - num_1_loss: 1.3172 - num_2_loss: 1.2526 - num_3_loss: 1.2202 - num_4_loss: 1.1979 - num_5_loss: 1.1644 - num_6_loss: 1.3128 - num_7_loss: 1.3094
Epoch 600/600 - loss: 8.7796 - num_1_loss: 1.3029 - num_2_loss: 1.2280 - num_3_loss: 1.2021 - num_4_loss: 1.2097 - num_5_loss: 1.2033 - num_6_loss: 1.3360 - num_7_loss: 1.2976
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
600 Epoch
PREDICT SRBIJA LOTO 7/39: [4, 9, x, y, z, 23, 37]
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
