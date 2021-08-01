import numpy as np

# TODO: Extend to other optimizer
#--- Keras functions

def neural_net(learning_rate=0.001, neurons=(64, 128, 64), act_func=('relu', 'relu', 'relu')):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, regularizers
    from keras import optimizers

    if(len(neurons) != len(act_func)):
        print("\t Error! Neurons and activations need to have same dimension")
        print("\t Error in Keras")
        exit()

    model = Sequential()
    model.add(Dense(units=neurons[0], input_dim=3, activation=act_func[0])) #, kernel_regularizer=regularizers.l2(0.01)))

    for i in range(len(neurons)-1):
        model.add(Dense(neurons[i+1], activation=act_func[0]))

    model.add(Dense(1, activation='linear'))

    #Setup optimizer
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = optimizers.SGD(lr=learning_rate)
    #Compile net
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    model.name = "NeuralNetworkRegressor_keras"

    print(model.summary())
    return model
