from pp2 import model
import numpy as np
from tqdm import tqdm
import h5py

all_data = np.load('simple_data.npz')
imgs_color = all_data['imgs']
imgs_color=imgs_color/255.0
a=all_data['aa']
b=all_data['bb']
c=all_data['cc']
d=all_data['dd']


# Train a little bit
nb_epoch = 10
mini_epoch = 10
num_steps = int(nb_epoch/mini_epoch)
for step in tqdm(range(0,num_steps)):
    h = model.fit(imgs_color,[a,b,c,d], batch_size = 32, nb_epoch=mini_epoch, verbose=1, validation_split=0.1, shuffle=True)
    model.save_weights('steer_comma_{0}_{1:4.5}.h5'.format(step,h.history['val_loss'][-1]),overwrite=True)
    print "hi"


all_data_test = np.load('simple_data_test.npz')
imgs_color_test = all_data_test['imgs']
imgs_color_test=imgs_color_test/255.0
a_test=all_data_test['aa']
b_test=all_data_test['bb']
c_test=all_data_test['cc']
d_test=all_data_test['dd']

preds = model.predict(imgs_color_test)
print preds



from scipy import misc
import numpy as np
import os
csv=open("stop-split2.csv")
f = open('workfile5.csv', 'w')
header = csv.readline().split(";")
print header
csv = csv.readlines()


i=0
for line in csv:
	fields = line.split(";")
	head, tail = os.path.split(fields[0])
	f.write(tail)
	f.write(";")
	f.write(str(int(preds[0][i])))
	f.write(";")
	f.write(str(int(preds[1][i])))
	f.write(";")
	f.write(str(int(preds[2][i])))
	f.write(";")
	f.write(str(int(preds[3][i])))
	f.write("\n")
	i=i+1
	print i
f.close()

