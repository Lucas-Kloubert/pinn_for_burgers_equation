import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import tensorflow as tf
from validation import tx_val, y_val, validation_fun

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for Burgers' equation on different combinations of hyperparameters using randomness
    Each parameter will only be tested once at most, so duplicate it if you want multiple combinations using it.
    The number of trained models will be equal to the lowest amount of values to test for an hyperparameter
    """
    
    # number of training samples
    num_train_samples = 1000
    # number of test samples
    num_test_samples = 1000
    # kinematic viscosity
    nu = 0.01


    ac_funs = ["tanh", "sigmoid", "relu",tf.keras.activations.selu,tf.keras.activations.elu] # Activation functions to be tested
    np.random.shuffle(ac_funs)  # Add randomness by shuffling
    nus = [nu/1, nu, nu*1, nu, nu]  # Different values of nu. Values should be duplicated because comparison doesn't always make sense between different values
    np.random.shuffle(nus)
    max_iters = [4000,5000,6000, 3000, 8000] # Maximum number of iterations to test (may be irrelevant if convergence is reached too quick)
    np.random.shuffle(max_iters)
    layers_sizes = [[20, 20, 20],[32, 16, 32],[16, 64, 16], [32,32,32], [12,28,44]] # Different structures for the hidden layer of the NN estimator
    np.random.shuffle(layers_sizes)

    # Create accumulators for training metrics so that comparative plots can be done between models
    iter_losses = []
    epoch_losses = []
    valid_losses = []

    nb_of_models = min([len(ac_funs), len(nus), len(max_iters), len(layers_sizes)])
    print("You can and should stop now if you set too many parameters to test \n Current Number of models to train = {}".format(nb_of_models))

    for index in range(nb_of_models):
        print("\n\nModel {} out of {}\n".format(index+1, nb_of_models))

        # build a core network model
        network = Network.build(activation=ac_funs[index], layers=layers_sizes[index])
        network.summary() # Comment this line if you have too many models to train or redundancy for layers_sizes
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
        lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, num_train_samples=num_train_samples, maxiter=max_iters[index], validation=validation_fun, x_valid=tx_val, y_valid=y_val, network=network) # You might want to not use validation if you train a large amount of models
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

        # plot u(t,x) distribution as a color-map
        fig = plt.figure(figsize=(7,4))
        gs = GridSpec(2, 3)
        plt.subplot(gs[0, :])
        plt.pcolormesh(t, x, u, cmap='rainbow')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        cbar.mappable.set_clim(-1, 1)
        # plot u(t=const, x) cross-sections
        t_cross_sections = [0.25, 0.5, 0.75]
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(gs[1, i])
            tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
            u = network.predict(tx, batch_size=num_test_samples)
            plt.plot(x_flat, u)
            plt.title('t={}'.format(t_cs))
            plt.xlabel('x')
            plt.ylabel('u(t,x)')
        plt.tight_layout()
        plt.show()


    # Pad the accumulators for each model horizontally so that comparisons are made easier on the plots
    max_iter = max([len(iter_loss_log) for iter_loss_log in iter_losses])
    max_epoch = max([len(epoch_loss_log) for epoch_loss_log in epoch_losses])

    for index, _ in enumerate(iter_losses):
        iter_losses[index]  += [(iter_losses[index][-1]) for _ in range(max_iter - len(iter_losses[index]))]
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