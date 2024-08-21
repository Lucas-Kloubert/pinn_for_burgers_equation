import scipy.optimize
import numpy as np
import tensorflow as tf

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.

    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: log metrics
        progbar: progress bar
    """

    def __init__(self, model, x_train, y_train, num_train_samples, factr=1e5, m=50, maxls=50, maxiter=5000, validation=False, network = None, x_valid = None, y_valid = None):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
            validation: optional argument, if not False, serves as the function used to compute validation loss. It must take a NN, inputs and corresponding ground-truth values as inputs and return predictions and validation loss.
            network: the sub-part of the model on which we want to evaluate the validation performance
        """

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.num_train_samples=num_train_samples

        self.validation = validation
        self.network = network
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.maxiter = maxiter
        self.metrics = ['loss']

        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':np.ceil(maxiter/num_train_samples), 'steps':np.ceil(num_train_samples), 'metrics':self.metrics})
        

        #Initialize class accumulators for useful metrics
        self.iter_loss = []
        self.epoch_loss = []
        self.valid_loss = []

    def set_weights(self, flat_weights):
        """
        Set weights to the model.

        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.

        Args:
            x: input data.

        Returns:
            loss and gradients for weights as tf.Tensor.
        """

        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.

        Args:
            weights: flatten weights.

        Returns:
            loss and gradients for weights as ndarray.
        """

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def callback(self, weights):
        """
        Callback that prints the progress to stdout.
        Also updates training-related accumulators.

        Args:
            weights: flatten weights.
        """
        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))
        self.iter_loss.append(loss) # Add a term to the accumulator for every iteration loss that was computed. Can be costly in terms of memory, comment this line when needed.
        self.epoch_loss[-1] += loss  # Sum iteration losses in the last term corresponding to this epoch

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        current_weights = initial_weights
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        # Iterate over epochs until either max_iter is reached or the model sufficiently converges 
        for epoch in range(int(np.ceil(self.maxiter/self.num_train_samples))):
            previous_iters = len(self.iter_loss)
            self.progbar.on_epoch_begin(epoch)
            self.epoch_loss.append(0) # Add a new term where the loss of the new epoch will be computed
            max_steps_in_this_epoch = min(np.ceil(self.num_train_samples),(self.maxiter-epoch*self.num_train_samples))
            _,_, dict = scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=current_weights,
            factr=self.factr, m=self.m, maxls=self.maxls, maxiter=max_steps_in_this_epoch,
            callback=self.callback) # Dict has an entry "warnflag" which is equal to 0 only when the training stopped based on convergence condition
            current_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ]) # Update weights for training
            self.epoch_loss[-1]/= (len(self.iter_loss)-previous_iters+1) # Normalize loss by the number of effective iterations this epoch
            self.progbar.on_epoch_end(1)

            # Compute validation only when asked to do so because it slows training
            if self.validation :
                _, valid_loss = self.validation(self.network, self.x_valid, self.y_valid)
                self.valid_loss.append(valid_loss)

            # Do not go through remaining epochs if the convergence condition was met
            if (dict["warnflag"] == 0 and epoch!=0) :
                break


        self.progbar.on_train_end()
        return self.iter_loss, self.epoch_loss, self.valid_loss     #Return accumulators so that the metrics can be plotted
