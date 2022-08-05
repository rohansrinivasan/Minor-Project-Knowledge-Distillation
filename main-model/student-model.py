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