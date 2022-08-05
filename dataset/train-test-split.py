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

#%% Single input model
n_length, n_features= X_train.shape[2],X_train.shape[3]
