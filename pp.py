from scipy import misc
import numpy as np
import os
csv=open("stop-split2.csv")
f = open('workfile', 'w')
header = csv.readline().split(";")
print header
csv = csv.readlines()
smcam = np.zeros((len(csv),3,960,1280),dtype=np.uint8)
a = np.zeros(len(csv))
b = np.zeros(len(csv))
c = np.zeros(len(csv))
d = np.zeros(len(csv))

i=0
for line in csv:
	fields = line.split(";")
	head, tail = os.path.split(fields[0])
	f.write(tail)
	image=misc.imread(fields[0])
	image=image.transpose((2,0,1))
	smcam[i]=image
	a[i]=int(fields[2])
	b[i]=int(fields[3])
	c[i]=int(fields[4])
	d[i]=int(fields[5])
	i=i+1
	print i



np.savez('simple_data_test.npz',
        imgs=smcam,
        aa=a,
        bb=b,
        cc=c,
        dd=d)