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