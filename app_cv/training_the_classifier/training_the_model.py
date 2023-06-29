import tensorflow as tf 
import cv2 
import os
import matplotlib.pyplot as plt 
import numpy as np
import random

Classes = ["0", "1", "2", "3","4", "5", "6"]

#data file is too large too be uplaoded in the git repositry
Datadirectory = "train/"

training_Data = [] 
def create_training_Data(Classes):
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num= Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array= cv2.imread (os.path.join(path, img))
                
                training_Data.append([img_array,class_num])
            except Exception as e:
                pass

create_training_Data(Classes)
random.shuffle(training_Data)
X = [] 
y = [] 
for features, label in training_Data:
    X.append(features)
    y.append(label)
    #X = np.array(X).reshape(-1, 48, 48, 3)

X = np.array(X).reshape(-1, 48, 48, 3)

X = X/255.0
Y = np.array(y)


from tensorflow.keras import datasets, layers, models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
earlystopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.0001,
                    patience=20,
                    verbose=1,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
)

history = model.fit(X, Y, epochs = 25, callbacks = earlystopping)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

model.save('../emotion_detector.h5')


