import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding, Input, merge, ELU
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2, activity_l2, l1
from keras.utils.np_utils import to_categorical
from keras import backend as K
import sklearn.metrics as metrics

# Define Model
nrows = 960
ncols = 1280
wr = 0.00001
dp = 0.

# video frame in,
frame_in = Input(shape=(3,nrows,ncols), name='img_input')
# convolution for image input
conv1 = Convolution2D(32,10,10,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
conv_l1 = conv1(frame_in)
Econv_l1 = ELU()(conv_l1)
pool_l1 = MaxPooling2D(pool_size=(2,2))(Econv_l1)

conv2 = Convolution2D(32,10,10,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
conv_l2 = conv2(pool_l1)
Econv_l2 = ELU()(conv_l2)
pool_l2 = MaxPooling2D(pool_size=(2,2))(Econv_l2)
drop_l2 = Dropout(dp)(pool_l2)

conv3 = Convolution2D(64,10,10,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
conv_l3 = conv3(drop_l2)
Econv_l3 = ELU()(conv_l3)
pool_l3 = MaxPooling2D(pool_size=(2,2))(Econv_l3)
drop_l3 = Dropout(dp)(pool_l3)

flat = Flatten()(drop_l3)

D1 = Dense(32,W_regularizer=l1(wr), init='lecun_uniform')(flat)
ED1 = ELU()(D1)
DED1 = Dropout(dp)(ED1)

S1 = Dense(64,W_regularizer=l1(wr), init='lecun_uniform')(DED1)
ES1 = ELU()(S1)

A = Dense(1, activation='linear', name='A', init='lecun_uniform')(ES1)
B = Dense(1, activation='linear', name='B', init='lecun_uniform')(ES1)
C = Dense(1, activation='linear', name='C', init='lecun_uniform')(ES1)
D = Dense(1, activation='linear', name='D', init='lecun_uniform')(ES1)

model = Model(input=frame_in, output=[A,B,C,D])
adam = Adam(lr=0.001)
model.compile(loss='mse',optimizer=adam)

