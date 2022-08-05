import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed,Conv1D,AveragePooling1D,Flatten,LSTM,Dense,BatchNormalization,Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical

from google.colab import drive
drive.mount("/content/drive")
file_path = r'/content/drive/MyDrive/HGR_DL/'
emg = np.load(file_path+'datagen2_emgDom.npy');
acc = np.load(file_path+'datagen2_accDom.npy');
gyr = np.load(file_path+'datagen2_gyrDom.npy');
y_true = np.load(file_path+'datagen2_y_true.npy');
y=np.vstack((y_true,range(0,len(y_true)))).transpose();
chN=3;ax=3; 
seglenE = 3000; #int(np.round(1.25*fs[0])); #number of samples to downsample to 
seglenA = 400; #int(np.round(1.25*fs[1])); #number of samples to downsample to 
n_steps=10;n_lengthE=300; n_lengthA=40;

# sEMG Signal
emg = emg.reshape((emg.shape[0],chN, seglenE));
emg1 = signal.resample(emg, seglenA, t=None, axis=2);
emg = np.transpose(emg,axes=(0,2,1));
emg1 = np.transpose(emg1,axes=(0,2,1));

# Accelerometer Signal
acc = acc.reshape((acc.shape[0],chN*ax, seglenA));
acc = np.transpose(acc,axes=(0,2,1));

# Gyroscope Signal
gyr = gyr.reshape((gyr.shape[0],chN*ax, seglenA));
gyr = np.transpose(gyr,axes=(0,2,1));

# Making a single feature matrix of all three modalities with 5000 samples of each

X = np.concatenate((emg1,acc,gyr),axis=2)

# Reshape
X = X.reshape((X.shape[0],n_steps, n_lengthA,chN+chN*ax*2));
emg = emg.reshape((emg.shape[0],n_steps, n_lengthE,chN));
acc = acc.reshape((acc.shape[0],n_steps, n_lengthA,chN*ax));
gyr = gyr.reshape((gyr.shape[0],n_steps, n_lengthA,chN*ax));

# KD CLASS
class Distiller1(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller1, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        
        super(Distiller1, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

# Reshaped data is split 70-30 where 70 is for the train split and 30 is for the test split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y_true)
train_idx = y_train[:,1];   y_train = y_train[:,0]
test_idx = y_test[:,1];     y_test = y_test[:,0]
emg_train = emg[train_idx,:,:,:]; 
emg_test = emg[test_idx,:,:,:];
acc_train = acc[train_idx,:,:,:]; 
acc_test = acc[test_idx,:,:,:];
gyr_train = gyr[train_idx,:,:,:]; 
gyr_test = gyr[test_idx,:,:,:];
del y, emg1, X,emg,acc,gyr

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_test1 = np.argmax(y_test, axis=1)

from keras.optimizers import adam_v2
n_outputs = y_train.shape[1]
verbose, epochs, batch_size = 1, 100, 64 #0, 15, 50
adam = adam_v2.Adam(lr=0.002)
es = EarlyStopping(monitor = 'val_accuracy',min_delta = 0.0002, patience = 5, verbose = 1,restore_best_weights = True)

#% Single input model
n_length, n_features= X_train.shape[2],X_train.shape[3]

# Base/Teacher Model
# The Base/Teacher Model used in our study is a multi-channel Deep convolutional network.

# Sequential Teacher Model 
model1 = Sequential()
model1.add(TimeDistributed(Conv1D(filters=6, kernel_size=7, activation='sigmoid'), input_shape=(None,n_length,n_features)))
model1.add(TimeDistributed(AveragePooling1D(pool_size=3)))
model1.add(TimeDistributed(Conv1D(filters=12, kernel_size=7, activation='sigmoid')))
model1.add(TimeDistributed(AveragePooling1D(pool_size=3)))
model1.add(TimeDistributed(Flatten()))
model1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model1.add(BatchNormalization(batch_size = batch_size))
model1.add(Dropout(0.2))
model1.add(Dense(n_outputs, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model1.summary()

#%
hist1 = model1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True, validation_data=(X_test, y_test)) #, epochs=epochs,callbacks=[es]
hist_arr1 = np.array([hist1.history['accuracy'],hist1.history['val_accuracy'],hist1.history['loss'],hist1.history['val_loss']])

## Model Result/Performance Plots
# Test Accuracy
plt.figure(1)
plt.plot(hist_arr1[1])
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)
plt.ylim(0.25,1.05)

# Validation Accuracy
plt.figure(2)
plt.plot(hist_arr1[0])
plt.ylabel('Train Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)
plt.ylim(0.25,1.05)

# Test Loss
plt.figure(3)
plt.plot(hist_arr1[3])
plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Loss
plt.figure(4)
plt.plot(hist_arr1[2])
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Student Model
# The Student Model used in our study is a similar multi-channel Deep convolutional network.
# The difference between the student model and teacher model is that the student model has much lesser parameters in comparison

# Sequential Student Model 
model2 = Sequential()
model2.add(TimeDistributed(Conv1D(filters=3, kernel_size=7, activation='sigmoid'), input_shape=(None,n_length,n_features)))
model2.add(TimeDistributed(AveragePooling1D(pool_size=3)))
model2.add(TimeDistributed(Conv1D(filters=6, kernel_size=7, activation='sigmoid')))
model2.add(TimeDistributed(AveragePooling1D(pool_size=3)))
model2.add(TimeDistributed(Flatten()))
model2.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model2.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model2.add(BatchNormalization(batch_size = batch_size))
model2.add(Dropout(0.2))
model2.add(Dense(n_outputs, activation='softmax'))
model2.summary()

#Distill Student to Teacher
distiller = Distiller1(student=model2, teacher=model1)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3,
)
hist6=distiller.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=True, validation_data=(X_test, y_test)) #, epochs=epochs,callbacks=[es]
hist_arr6 = np.array([hist6.history['accuracy'],hist6.history['val_accuracy'],hist6.history['student_loss'],hist6.history['distillation_loss'],hist6.history['val_accuracy'],hist6.history['val_student_loss']])


hist3=distiller.fit(X_train, y_train, epochs=epochs) #, epochs=epochs,callbacks=[es]
# Test Accuracy
plt.figure(5)
plt.plot(hist_arr2[0])
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Accuracy
plt.figure(2)
plt.plot(hist_arr2[3])
plt.ylabel('Train Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Test Loss
plt.figure(3)
plt.plot(hist_arr2[1])
plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Loss
plt.figure(4)
plt.plot(hist_arr2[4])
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Distillation Loss
plt.figure(4)
plt.plot(hist_arr2[2])
plt.ylabel('Teacher over Student Distillation Loss')
plt.xlabel('Epoch')
plt.grid(True)

# # Set teacher model type to .h5 and save to acquire size
import tempfile
import os
_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model1, keras_file, include_optimizer=False)
print('Saved base model to:', keras_file)

# Set student model type to .h5 and save to acquire size
import tempfile
import os
_, KD_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model2, KD_keras_file, include_optimizer=False)
loaded_2 = keras.models.load_model("my_model")
print('Saved Distilled Keras model to:', KD_keras_file)

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

  print("Size of Base Model: %.2f bytes" % (get_gzipped_model_size(keras_file)))

  print("Size of Distilled Model: %.2f bytes" % (get_gzipped_model_size(KD_keras_file)))

# Base Model Accuracy
y1_pred = model1.evaluate(X_test, y_test, verbose=0)[1]
print("Base Model Accuracy:","{:.3f}%".format(y1_pred*100))

# Distilled Model Accuracy
y2_pred = distiller.evaluate(X_test, y_test, verbose=0)[1]
print("Distilled Model Accuracy","{:.3f}%".format(y2_pred*100))