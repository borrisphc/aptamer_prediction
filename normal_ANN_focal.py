import pandas as pd
import numpy as np
import sys
from tensorflow.python.keras import backend as K



Model_names = sys.argv[7]
protein_train_path = sys.argv[1]
protein_test_path = sys.argv[2]
dna_train_path = sys.argv[3]
dna_test_path = sys.argv[4]
y_train_path = sys.argv[5]
y_test_path = sys.argv[6]


def fun(fn):
    test = pd.read_csv(fn)
    test = test[test.columns[1:]]
    return np.array(test)

def funy(fn):
    uu = pd.read_csv(fn)
    return np.array((uu["x"]))

Protein_train = fun(protein_train_path)
Protein_test = fun(protein_test_path)
DNA_train = fun(dna_train_path)
DNA_test = fun(dna_test_path)

y_train = funy(y_train_path)
y_test = funy(y_test_path)


# BatchNormalization
from keras.layers import Embedding
from keras.layers import Flatten, Dense
from keras.layers import SimpleRNN
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dropout
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras import layers
from keras import Input
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.utils import multi_gpu_model
from keras.optimizers import Adam, SGD, RMSprop




def focal_loss(alpha=0.65,gamma= 2 ):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy











DNA_input = Input( shape = (300,))
DNA_layer = Dense( units= 2048,input_shape = (300,) , activation="relu") (DNA_input)
DNA_layer = Dense( units= 1024 , activation="relu") (DNA_layer)
DNA_layer = Dense( units= 512 , activation="relu") (DNA_layer)
#DNA_layer = Dense( units= 512 , activation="relu") (DNA_layer)
#DNA_layer = Dropout( 0.02) (DNA_layer)
DNA_layer = Dense( units=256 , activation="relu") (DNA_layer)
#DNA_layer = Dropout( 0.03) (DNA_layer)
DNA_layer = Dense( units=128 , activation="relu") (DNA_layer)
#DNA_layer = Dense( units=128 , activation="relu") (DNA_layer)
#DNA_layer = Dense( units=128 , activation="relu") (DNA_layer)
#DNA_layer = Dense( units=32 , activation="relu") (DNA_layer)


Protein_input =  Input( shape = (2000,))

Protein_layer = Dense( units= 2048,input_shape = (2000,) , activation="relu") (Protein_input) 
Protein_layer = Dense( units= 1024 , activation="relu") (Protein_layer)
Protein_layer = Dense( units= 512 , activation="relu") (Protein_layer)
#Protein_layer = Dropout( 0.02) (Protein_layer)
Protein_layer = Dense( units=256 , activation="relu") (Protein_layer)
#Protein_layer = Dropout( 0.03) (Protein_layer)
Protein_layer = Dense( units=128 , activation="relu") (Protein_layer)
#Protein_layer = Dense( units=128 , activation="relu") (Protein_layer)
#Protein_layer = Dense( units=128 , activation="relu") (Protein_layer)
#Protein_layer = Dense( units=32 , activation="relu") (Protein_layer)

concatenated = layers.concatenate([Protein_input,DNA_input])
#concatenated = layers.concatenate([DNA_layer,Protein_layer])
#concatenated = Dropout( 0.01)(concatenated)
#concatenated = Dense(2048, activation="relu")(concatenated)
#concatenated = Dropout( 0.1)(concatenated)
concatenated = Dropout( 0.01)(concatenated)
concatenated = Dense(128, activation="relu")(concatenated)
concatenated = Dropout( 0.01)(concatenated)
concatenated = Dense(32, activation="relu")(concatenated)
concatenated = Dense(32, activation="relu")(concatenated)
Out_put = Dense(1, activation="sigmoid")(concatenated)
model = Model( [DNA_input, Protein_input], Out_put)
opt = Adam(lr=1e-4)
model.compile( optimizer= opt, loss=focal_loss(), metrics = ['acc'])



callbacks_list = [
    EarlyStopping( monitor = "acc", # 如果acc都沒有變就停止
                    patience = 2900
                    ), # 多久沒有動就停
    ModelCheckpoint( filepath = Model_names + '.h5', # 存的檔名
                     monitor = 'acc',
                     save_best_only = True ),
    ReduceLROnPlateau( monitor = 'acc',
                       factor = 0.1 , # 減少的幅度，變成原先的十分之一
                       patience = 20 )#, # 多久沒動就開始變學習速度
    #TensorBoard( log_dir = "/src/notebooks/seed1024/my_log_dir",
    #             histogram_freq = 2,
      #           embeddings_freq = 2
#)
]

history = model.fit( [DNA_train, Protein_train], y_train, epochs = 3000, batch_size= 256, 
          validation_data=([DNA_test, Protein_test],y_test),
          callbacks = callbacks_list
         )




import pandas as pd

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
# or save to csv: 
last_val_acc = str(round(hist_df.val_acc.values[len(hist_df.val_acc)-1],3))
last_acc = str(round(hist_df.acc.values[len(hist_df.val_acc)-1],3))
hist_csv_file = "history_val_acc_%s_acc_%s_id_%s.csv"%( last_val_acc, last_acc, Model_names )
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
history_dict = history.history
acc_value = history_dict['acc']
val_acc_values = history.history['val_acc']


epochs = range(1,len(acc_value)+1 )
plt.plot(epochs, acc_value, 'b', label = "Training acc")
plt.plot(epochs, val_acc_values, 'b', label  = "Validation acc", color= "red")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.savefig( "history_val_acc_%s_acc_%s_id_%s.png"%( last_val_acc, last_acc, Model_names ))
plt.clf()

