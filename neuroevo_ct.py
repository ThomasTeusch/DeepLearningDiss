import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, argument_default=argparse.SUPPRESS)
parser.add_argument('--modus', 		type=str,	default="fitness",    	help='Modus of python file: write_pop, write_fit, evo')
parser.add_argument('--fittrain', 	type=float,	default=np.inf,      	help='Fitness train of parent')
parser.add_argument('--fittest',  	type=float, 	default=np.inf,       	help='Fitness test of parent')
parser.add_argument('--epoch',    	type=int, 	default=1,            	help='epoch of neuroevo')
parser.add_argument('--bestpop1',	type=str, 	default="0000000000",	help='String of best population1')
parser.add_argument('--bestpop2', 	type=str, 	default="0000000000",	help='String of last population2')
parser.add_argument('--actpop', 	type=str, 	default="0000000000",	help='String of actual population')
parser.add_argument('--sigma',  	type=float, 	default=0.1,       	help='Sigma of mutation')
args = parser.parse_args()

def arrToString(bitcode):
    s = ''
    for i in bitcode: s= s + str(i)
    return s

def stringToArr(pop):
    return np.array([int(i) for i in pop])
 
def mutation(bitcode, sigma):
    return np.array([(bit+1)%2 if np.random.random()<sigma else bit for bit in bitcode])

def crossover(bitcode1, bitcode2):
    part11 = bitcode1[0:5] ; part12 = bitcode1[5:10]
    part21 = bitcode2[0:5] ; part22 = bitcode2[5:10]

    cross1=np.zeros(10,dtype=int) ; cross2=np.zeros(10,dtype=int)

    cross1[0:5]=part11 ; cross1[5:10]=part22
    cross2[0:5]=part21 ; cross2[5:10]=part12
    return cross1, cross2

def bitTOParam(bitcode):
    # First two digits: Number of layers l ("2" "3" "4" "5")
    # Next two digits: Number of neurons n ("32" "64" "128" "256")
    # Next three digits: Activation function a ("selu" "relu" "tanh" "celu" "rrelu" "leaky_relu" "sigmoid" "tanhshrink")
    # Next digit: Loss function f ("mse" "mae") 
    # Last two digits: Dropout rate d ("0.00" "0.01" "0.05" "0.10")
    #print("bitcode")
    #print(bitcode)
    l = bitcode[0:2] ; n = bitcode[2:4] ; a = bitcode[4:7] ; f = bitcode[7:8] ; d = bitcode[8:10]
    #print("l, n, a, f, d")   
    #print(l, n, a, f, d)   
    #l:
    if   ( np.array_equal(l, np.array([0,0])) ): layer="2"
    elif ( np.array_equal(l, np.array([0,1])) ): layer="3"
    elif ( np.array_equal(l, np.array([1,0])) ): layer="4"
    elif ( np.array_equal(l, np.array([1,1])) ): layer="5"
    else: print("Error in number of layer!") ; exit
    
    #n
    if   ( np.array_equal(n, np.array([0,0])) ): neurons="32"
    elif ( np.array_equal(n, np.array([0,1])) ): neurons="64"
    elif ( np.array_equal(n, np.array([1,0])) ): neurons="128"
    elif ( np.array_equal(n, np.array([1,1])) ): neurons="256"
    else: print("Error in number of neurons!") ; exit
    
    #a
    if   ( np.array_equal(a, np.array([0,0,0])) ): actf="selu"
    elif ( np.array_equal(a, np.array([0,0,1])) ): actf="relu"
    elif ( np.array_equal(a, np.array([0,1,0])) ): actf="tanh"
    elif ( np.array_equal(a, np.array([0,1,1])) ): actf="celu"
    elif ( np.array_equal(a, np.array([1,0,0])) ): actf="rrelu"
    elif ( np.array_equal(a, np.array([1,0,1])) ): actf="leaky_relu"
    elif ( np.array_equal(a, np.array([1,1,0])) ): actf="sigmoid"
    elif ( np.array_equal(a, np.array([1,1,1])) ): actf="tanhshrink"
    else: print("Error in activation function!") ; exit
    
    #f
    if   ( np.array_equal(f, np.array([0])) ): loss="mse"
    elif ( np.array_equal(f, np.array([1])) ): loss="mae"
    else: print("Error in loss function!") ; exit
    
    #d:
    if   ( np.array_equal(d, np.array([0,0])) ): drop="0.00"
    elif ( np.array_equal(d, np.array([0,1])) ): drop="0.01"
    elif ( np.array_equal(d, np.array([1,0])) ): drop="0.05"
    elif ( np.array_equal(d, np.array([1,1])) ): drop="0.10"
    else: print("Error in dropout rate!") ; exit

    return layer, neurons, actf, loss, drop

# Length of bitstring
N = 10 

sigma = args.sigma

# Write population to file
if(args.modus.lower()=="write_pop"):
    foutput = open('results_neuroevo_ct.out',"a+")
    layer, neurons, actf, loss, drop = bitTOParam(stringToArr(args.actpop))
    foutput.write("{0:>10s}\t{1:>1s}\t{2:>3s}\t{3:>10s}\t{4:>3s}\t{5:>3s}\t".format(args.actpop, layer, neurons, actf, loss, drop))
    foutput.close()

# Write fitness of population to file
elif(args.modus.lower()=="write_fit"):
#    if((args.fittrain < np.inf) and (args.fittest < np.inf)):
    foutput = open('results_neuroevo_ct.out',"a+")
    if((args.fittrain < 10) and (args.fittest < 10)):
        foutput.write("{0:>3.9f}\t{1:>3.9f} \n".format(args.fittrain, args.fittest) )
        foutput.close()
    else:
        foutput.write("{0:>3.9f}\t{1:>3.9f} \n".format(9.999999999, 9.999999999))
        foutput.close()
    exit()

elif(args.modus.lower()=="evo"):
#Initialize population
    if(args.epoch < 1): 	
        x1 = np.random.randint(2, size=N)
        x2 = np.random.randint(2, size=N)

# Read last populations from file:
# Important to check whether population has
# already been calculated
    else:
        f = open('results_neuroevo_ct.out', "r")
        data = [[c for c in line.split()] for line in f.readlines()]
        f.close()
        data = np.array(data)
        data=([i[0] for i in data])
        s=[]
        for i in range(len(data)):
            s=np.append(s, [data[i]])
        data = np.reshape(s,(len(s),1))
        s=np.array([])
    
        for i in range(len(data)):
            tmp = np.array(stringToArr(data[i][0]))
            s = np.append(s, tmp, axis=-1)
        data = np.reshape(s,(i+1,int(len(s)/(i+1))))
    
    # crossover (recombination) algorithm (not for (1+1)-EA)
    if(args.epoch > 0):
        x1 = stringToArr(args.bestpop1)
        x2 = stringToArr(args.bestpop2)
    
        x1, x2 = crossover(x1, x2)
    
    # mutation algorithm: If population already existent create new population
    if(args.epoch > 0):
        match1=True
        while match1:
            match1=False
            newpop1 = mutation(x1, sigma)
            for i in range(len(data)):
                if( np.array_equal(data[i], newpop1) ): 
                    match1=True
    
        match2=True
        while match2:
            match2=False
            newpop2 = mutation(x2, sigma)
            for i in range(len(data)):
                if( np.array_equal(data[i], newpop2) ):
                    match2=True
    
    else:
        newpop1 = x1
        newpop2 = x2

    ftmp = open('tmp_pop_ct',"w+")
    layer, neurons, actf, loss, drop = bitTOParam(newpop1)
    ftmp.write("{0:>10s}\t{1:>1s}\t{2:>3s}\t{3:>10s}\t{4:>3s}\t{5:>3s}\n".format(arrToString(newpop1), layer, neurons, actf, loss, drop))
    layer, neurons, actf, loss, drop = bitTOParam(stringToArr(newpop2))
    ftmp.write("{0:>10s}\t{1:>1s}\t{2:>3s}\t{3:>10s}\t{4:>3s}\t{5:>3s}\n".format(arrToString(newpop2), layer, neurons, actf, loss, drop))
    ftmp.close()

