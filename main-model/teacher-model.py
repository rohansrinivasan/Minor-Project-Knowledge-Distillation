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

#%%
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