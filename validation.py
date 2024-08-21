import numpy as np

t_validation = [0.4, 0.6, 1, 3]

tx_val = [[t_validation[i%4], 1-0.25*(i//4+1)] for i in range(12)]


tx_val += [[1, 1-0.1*(i+1)] for i in range(9)]

y_val = [0.34229, 0.26902, 0.18817, 0.07511, 0.66797, 0.53211, 0.37500, 0.15018, 0.93680, 0.77724, 0.55833, 0.22485, 0.07538, 0.15065, 0.22567, 0.30021, 0.37442, 0.44782, 0.52027, 0.59148, 0.66002]

assert(len(tx_val)==len(y_val))

def validation_fun (network, x, y):
    valid_loss = 0
    outputs = []
    for i, item in enumerate(x):
        output = network.predict([item])
        outputs.append(output[0])
        valid_loss += np.abs(output + y[i]) #For now loss is the direct sum of absolute values of the errors
    valid_loss = valid_loss[0][0] / len(y_val) #We normalize based on length of the validation set
    return outputs, valid_loss
