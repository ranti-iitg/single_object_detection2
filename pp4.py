from scipy import misc
import numpy as np
import os
csv=open("stop-split2.csv")
f = open('workfile2.csv', 'w')
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

