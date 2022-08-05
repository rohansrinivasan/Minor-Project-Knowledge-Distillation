# # Set teacher model type to .h5 and save to acquire size
import tempfile
import os
_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model1, keras_file, include_optimizer=False)
print('Saved basel model to:', keras_file)

# Set student model type to .h5 and save to acquire size
import tempfile
import os
_, KD_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model2, KD_keras_file, include_optimizer=False)
loaded_2 = keras.models.load_model("my_model")
print('Saved Distilled Keras model to:', KD_keras_file)