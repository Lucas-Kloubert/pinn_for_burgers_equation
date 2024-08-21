import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
from validation import tx_val, y_val, validation_fun

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for Burgers' equation on different combinations of hyperparameters using GridSearch
    Every combination of parameters will be tried given the lists of values for each hyperparameter.
    Beware: since models are trained successively, this can quickly become a very long process with unreadable plots (especially on CPU)
    """
    
    # number of training samples
    num_train_samples = 1000
    # number of test samples
    num_test_samples = 1000
    # kinematic viscosity
    nu = 0.01


    ac_funs_diff = ["tanh", "sigmoid"] # Activation functions to be tested
    nus_diff = [nu] # Different values of nu.
    max_iters_diff = [4000, 3000] # Maximum number of iterations to test (may be irrelevant if convergence is reached too quickly)
    layers_sizes_diff = [[32, 16, 32],[12,28,44]] # Different structures for the hidden layer of the NN estimator

    # Number of models to be trained
    tot_combinations = len(ac_funs_diff)*len(nus_diff)*len(max_iters_diff)*len(layers_sizes_diff)

    # Prepare the complete arrays of hyperarameters values to be iterated on
    ac_funs = [ac_funs_diff[i % len(ac_funs_diff)] for i in range(tot_combinations)]
    nus = [nus_diff[(i // len(ac_funs_diff)) % len(nus_diff)] for i in range(tot_combinations)]
    max_iters = [max_iters_diff[(i // (len(ac_funs_diff)*len(nus_diff))) % len(max_iters_diff)] for i in range(tot_combinations)]
    layers_sizes = [layers_sizes_diff[(i // (len(ac_funs_diff)*len(nus_diff)*len(max_iters_diff))) % len(layers_sizes_diff)] for i in range(tot_combinations)]


    #print(np.column_stack((ac_funs, nus, max_iters, layers_sizes)))

    print("You can and should stop now if you set too many parameters to test \n Current Number of models to train = {}".format(tot_combinations))

    
    iter_losses = []
    epoch_losses = []
    valid_losses = []

    for index in range(len(ac_funs)):
        print("\n\nModel {} out of {}\n".format(index+1, tot_combinations))


        # build a core network model
        network = Network.build(activation=ac_funs[index], layers=layers_sizes[index])
        # build a PINN model
        pinn = PINN(network, nus[index]).build()

        # create training input
        tx_eqn = np.random.rand(num_train_samples, 2)          # t_eqn =  0 ~ +1
        tx_eqn[..., 1] = 2 * tx_eqn[..., 1] - 1                # x_eqn = -1 ~ +1
        tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1  # x_ini = -1 ~ +1
        tx_ini[..., 0] = 0                                     # t_ini =  0
        tx_bnd = np.random.rand(num_train_samples, 2)          # t_bnd =  0 ~ +1
        tx_bnd[..., 1] = 2 * np.round(tx_bnd[..., 1]) - 1      # x_bnd = -1 or +1

        # create training output
        u_eqn = np.zeros((num_train_samples, 1))               # u_eqn = 0
        u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])    # u_ini = -sin(pi*x_ini)
        u_bnd = np.zeros((num_train_samples, 1))               # u_bnd = 0

        # train the model using L-BFGS-B algorithm
        x_train = [tx_eqn, tx_ini, tx_bnd]

        y_train = [ u_eqn,  u_ini,  u_bnd]
        lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, num_train_samples=num_train_samples, maxiter=max_iters[index], validation=validation_fun, x_valid=tx_val, y_valid=y_val, network=network)  # You might want to not use validation if you train a large amount of models
        iter_loss, epoch_loss, valid_loss = lbfgs.fit()

        iter_losses.append(iter_loss)
        epoch_losses.append(epoch_loss)
        valid_losses.append(valid_loss)

        # predict u(t,x) distribution
        t_flat = np.linspace(0, 1, num_test_samples)
        x_flat = np.linspace(-1, 1, num_test_samples)
        t, x = np.meshgrid(t_flat, x_flat)
        tx = np.stack([t.flatten(), x.flatten()], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)
        u = u.reshape(t.shape)


    # Pad the accumulators for each model horizontally so that comparisons are made easier on the plots
    max_iter = max([len(iter_loss_log) for iter_loss_log in iter_losses])
    max_epoch = max([len(epoch_loss_log) for epoch_loss_log in epoch_losses])

    for index, _ in enumerate(iter_losses):
        iter_losses[index] += [(iter_losses[index][-1]) for _ in range(max_iter - len(iter_losses[index]))]
        epoch_losses[index] += [((epoch_losses[index])[-1]) for _ in range(max_epoch - len(epoch_losses[index]))]
        valid_losses[index] += [((valid_losses[index])[-1]) for _ in range(max_epoch - len(valid_losses[index]))]

    assert(len(iter_losses[index])==max_iter)

    # Comparative plot for iteration lossses
    fig = plt.figure(figsize=(7,4))
    for i, iter_loss in enumerate(iter_losses):
        plt.plot(iter_loss, label = "ac_fun : {}, nu = {}, max_iter = {}, layer_size = {}".format(ac_funs[i], nus[i], max_iters[i], layers_sizes[i]))
    plt.title("Loss over iterations")
    plt.legend()
    plt.show()

    # Comparative plot for losses over successive epochs
    fig = plt.figure(figsize=(7,4))
    for i, iter_loss in enumerate(iter_losses):
        plt.plot(epoch_losses[i], label = "ac_fun : {}, nu = {}, max_iter = {}, layer_size = {}".format(ac_funs[i], nus[i], max_iters[i], layers_sizes[i]))
    plt.title("Loss over epochs")
    plt.legend()
    plt.show()

    # Comparative plot for validation lossses
    fig = plt.figure(figsize=(7,4))
    for i, valid_loss in enumerate(valid_losses):
        plt.plot(valid_loss, label = "ac_fun : {}, nu = {}, max_iter = {}, layer_size = {}".format(ac_funs[i], nus[i], max_iters[i], layers_sizes[i]))
    plt.title("Validation Loss over epochs")
    plt.legend()
    plt.show()
