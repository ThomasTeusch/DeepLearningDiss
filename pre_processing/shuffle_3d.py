#=========================================
# shuffle PES
#=========================================
import sys
import random
import numpy as np
np.set_printoptions(9)

#-----------
#--- input file as argument:
if ( len(sys.argv)<2 ):
  print ()
  print ("ERROR! File name not given. Aborting.")
  print ()
  exit()

s_file = sys.argv[1]

#--- generate seed:
seed = 23
np.random.seed(seed)

#--- open input file:
f_in = open(s_file, "r")
data = [[np.float64(c) for c in line.split()] for line in f_in.readlines()]
f_in.close()

#--- shuffle:
X = np.array(data, dtype = np.float64)
np.random.shuffle(X)

#--- write to output file:
f_out = open("%s_shuff%s.dat" % (s_file[:-4],seed), "w+")

for i in range(len(X)):
    #f_out.write("{0:8.2f} {1:>8.2f} {2:>8.2f} {3:>16.8f} {4:>16.8f} {5:>6.2f}\n".format(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5]))
    f_out.write("{0:8.2f} {1:>8.2f} {2:>8.2f} {3:>16.8f} {4:>3.2f}\n".format(X[i][0], X[i][1], X[i][2], X[i][3], 1.00))  # weight set on 1.00 per default

f_out.close()
