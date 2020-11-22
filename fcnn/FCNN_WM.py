import os
import json
import time
import numpy as np

class FCNN_WM(object):
    '''
            Description: Class to define a Fully Connected Neural Network (FCNN)
                         with weight mirrors (WM)
    '''

    def __init__(self, sizes, save_dir):
        '''
        Description: initialize the biases, forward weights and backward weights using
        a Gaussian distribution with mean 0, and variance 1.
        Params:
            - sizes: a list of size L; where L is the number of layers
                     in the deep neural network and each element of list contains
                     the number of neuron in that layer.
                     first and last elements of the list corresponds to the input
                     layer and output layer respectively
                     intermediate layers are hidden layers.
            - save_dir: the directory where all the data of experiment will be saved
        '''
        self.num_layers = len(sizes)
        self.save_dir = save_dir
        # setting appropriate dimensions for weights and biases
        self.biases = [np.sqrt(1. / (x + y)) * np.random.randn(y, 1)
                       for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.sqrt(1. / (x + y)) * np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.backward_weights = [np.sqrt(1. / (x + y)) * np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        # define the variables to save data in during training and testing
        self.data = {}
    
    def print_and_log(self, log_str):
        '''
        Description: Print and log messages during experiments
        Params:
            - log_str: the string to log
        '''
        print(log_str)
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f_:
            f_.write(log_str + '\n')
    
    def sigmoid(self, out):
        '''
        Description: the sigmoid activation function
        Params:
            - out: a list or a matrix to perform the activation on
        Outputs: the sigmoid activated list or a matrix
        '''
        return 1.0 / (1.0 + np.exp(-out))

    def delta_sigmoid(self, out):
        '''
        Description: the derivative of sigmoid activation function
        Params:
            - out: a list or a matrix to perform the activation on
        Outputs: the sigmoid prime activated list or matrix
        '''
        return self.sigmoid(out) * (1 - self.sigmoid(out))

    def SigmoidCrossEntropyLoss(self, a, y):
        """
        Description: the cross entropy loss
        Params:
            - a: the last layer activation
            - y: the target one hot vector
        Outputs: a loss value
        """
        return np.mean(np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)), axis=0))

    def feedforward(self, x):
        '''
        Description: Forward Passes an image feature matrix through the Deep Neural
                                 Network Architecture.
        Params:
            - x: the input signal
        Outputs: 2 lists which stores outputs and activations at every layer:
                 the 1st list is non-activated and 2nd list is activated
        '''
        activation = x
        activations = [x]  # list to store activations for every layer
        outs = []          # list to store out vectors for every layer
        for b, w in zip(self.biases, self.weights):
            out = np.matmul(w, activation) + b
            outs.append(out)
            activation = self.sigmoid(out)
            activations.append(activation)

        return outs, activations

    def get_batch(self, X, y, batch_size):
        '''
        Description: A data iterator for batching of input signals and labels
        Params::
            - X, y: lists of input signals and its corresponding labels
            - batch_size: size of the batch
        Outputs: a batch of input signals and labels of size equal to batch_size
        '''
        for batch_idx in range(0, X.shape[0], batch_size):
            batch = (X[batch_idx:batch_idx + batch_size].T,
                     y[batch_idx:batch_idx + batch_size].T)
            yield batch

    def backpropagate(self, x, y, eval_delta_angle=False):
        '''
        Description: Based on the derivative(delta) of cost function the gradients(rate of change
                     of cost function with respect to weights and biases) of weights and biases are calculated.
                     The variables del_b and del_w are of same size as all the forward weights and biases
                     of all the layers. The variables del_b and del_w contains the gradients which
                     are used to update the forward weights and biases.
        Params:
            - x, y: training feature and corresponding label
            - eval_delta_angle: a boolean to determine if the angle between deltas should be computed
        Outputs:
            - del_b: gradient of bias
            - del_w: gradient of weight
        '''
        # Set a variable to store angle during evaluation only
        if eval_delta_angle:
            deltas_angles = {}

        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        outs, activations = self.feedforward(x)

        # Cost function
        loss = self.SigmoidCrossEntropyLoss(activations[-1], y)

        # calculate derivative of cost Sigmoid Cross entropy which is to be minimized
        delta_cost = activations[-1] - y
        # backward pass to reduce cost gradients at output layers
        delta = delta_cost
        del_b[-1] = np.expand_dims(np.mean(delta, axis=1), axis=1)
        del_w[-1] = np.matmul(delta, activations[-2].T)

        # updating gradients of each layer using reverse or negative indexing, by propagating
        # gradients of previous layers to current layer so that gradients of weights and biases
        # at each layer can be calculated
        for l in range(2, self.num_layers):
            out = outs[-l]
            delta_activation = self.delta_sigmoid(out)
            if eval_delta_angle:
                # compute both FA and BP deltas and the angle between them
                delta_bp = np.matmul(self.weights[-l + 1].T, delta) * delta_activation
                delta = np.matmul(self.backward_weights[-l + 1], delta) * delta_activation
                deltas_angles['layer_{}'.format(self.num_layers - l)] = self.angle_between(delta_bp, delta)
            else:
                delta = np.matmul(self.backward_weights[-l + 1], delta) * delta_activation
            del_b[-l] = np.expand_dims(np.mean(delta, axis=1), axis=1)
            del_w[-l] = np.dot(delta, activations[-l - 1].T)
        if eval_delta_angle:
            return deltas_angles
        else:
            return loss, del_b, del_w

    def angle_between(self, A, B):
        '''
        Description: computes the angle between two matrices A and B
        Params:
            - A: a first matrix
            - B: a second matrix
        Outputs:
            - angle: the angle between the two vectors resulting from vectorizing and normalizing A and B
        '''
        flat_A = np.reshape(A, (-1))
        normalized_flat_A = flat_A / np.linalg.norm(flat_A)

        flat_B = np.reshape(B, (-1))
        normalized_flat_B = flat_B / np.linalg.norm(flat_B)

        angle = (180.0 / np.pi) * np.arccos(np.clip(np.dot(normalized_flat_A, normalized_flat_B), -1.0, 1.0))
        return angle

    def evaluate_angles(self, X_train, y_train):
        '''
        Description: computes the angle between both:
                        - the forward and backwards matrices
                        - the delta signals
        Params:
            - X_train, y_train: training feature and corresponding label
        Outputs:
            - deltas_angles: the angle between the delta signal and the backpropagation delta signal
            - weights_angles: the angle between the forward and backwards matrices
        '''

        # Evaluate angles between matrices
        weights_angles = {}
        for layer, (w, back_w) in enumerate(zip(self.weights, self.backward_weights)):
            matrix_angle = self.angle_between(w.T, back_w)
            weights_angles['layer_{}'.format(layer)] = matrix_angle
            log_str = 'In layer {} angle between matrices: {}'.format(self.num_layers - layer, matrix_angle)
            self.print_and_log(log_str)

        # Evaluate angles between delta signals
        [sample_x, sample_y] = list(next(self.get_batch(X_train, y_train, batch_size=1)))
        deltas_angles = self.backpropagate(sample_x, sample_y, eval_delta_angle=True)
        log_str = 'Angle between deltas: {}'.format(deltas_angles)
        self.print_and_log(log_str)
        return deltas_angles, weights_angles

    def eval(self, X, y):
        '''
        Description: Based on trained(updated) weights and biases, predict a batch of labels and compare
                     them with the original labels and calculate accuracy
        Params:
            - X: test input signals
            - y: test labels
        Outputs: accuracy of prediction
        '''
        outs, activations = self.feedforward(X.T)
        # count the number of times the postion of the maximum value is the predicted label
        count = np.sum(np.argmax(activations[-1], axis=0) == np.argmax(y.T, axis=0))
        test_accuracy = 100. * count / X.shape[0]
        return test_accuracy


    def mirror(self, batch_size, X_shape=None, mirror_learning_rate=0.01, noise_amplitude=0.1):
        '''
        Description: weight mirroring by feeding an iid Gaussian noise through *each* layer of the network
                     If the iid Gaussian noise is generated once and get forward-propagated,
                     the iid property is lost for hidden layers.
        Params:
            - batch_size: size of the mirroring batch
            - X_shape: the shape of the noise matrix
            - mirror_learning_rate: eta which controls the size of changes in backward weights
            - noise_amplitude: the amplitude of the iid Gaussian noise
        '''
        if not X_shape is None:
            n_batches = int(X_shape[0] / batch_size)
        else:
            n_batches = 1

        for i in range(n_batches):
            for layer, (b, w, back_w) in enumerate(zip(self.biases, self.weights, self.backward_weights)):
                noise_x = noise_amplitude * (np.random.rand(w.shape[1], batch_size) - 0.5)
                noise_y = self.sigmoid(np.matmul(w, noise_x) + b)
                # update the backward weight matrices using the equation 7 of the paper manuscript
                back_w += mirror_learning_rate * np.matmul(noise_x, noise_y.T)

        # Prevent feedback weights growing too large
        for layer, (b, w, back_w) in enumerate(zip(self.biases, self.weights, self.backward_weights)):
            x = np.random.rand(back_w.shape[1], batch_size)
            y = np.matmul(back_w, x)
            y_std = np.mean(np.std(y, axis=0))
            back_w = 0.5 * back_w / y_std

    def train(self, X_train, y_train, X_test, y_test, batch_size, learning_rate, epochs, test_frequency):
        '''
        Description: Batch-wise trains image features against corresponding labels.
                     The forward and backward weights and biases of the neural network are updated through
                     the Kolen-Pollack algorithm on batches using SGD
                     del_b and del_w are of same size as all the forward weights and biases
                     of all the layers. del_b and del_w contains the gradients which
                     are used to update forward weights and biases

        Params:
            - X_train, y_train: lists of training features and corresponding labels
            - X_test, y_test: lists of testing features and corresponding labels
            - batch_size: size of the batch
            - learning_rate: eta which controls the size of changes in weights & biases
            - epochs: no. of times to iterate over the whole data
            - test_frequency: the frequency of the evaluation on the test data
        '''
        n_batches = int(X_train.shape[0] / batch_size)

        # Start with an initial update of the backward matrices with weight mirroring
        self.mirror(batch_size, X_train.shape)

        for j in range(epochs):
            # initialize the epoch field in the data to store
            self.data['epoch_{}'.format(j)] = {}
            start = time.time()
            epoch_loss = []
            batch_iter = self.get_batch(X_train, y_train, batch_size)

            for i in range(n_batches):
                (batch_X, batch_y) = next(batch_iter)
                batch_loss, delta_del_b, delta_del_w = self.backpropagate(batch_X, batch_y)
                epoch_loss.append(batch_loss)
                del_b = delta_del_b
                del_w = delta_del_w
                # update weight and biases
                self.weights = [w - (learning_rate / batch_size)
                                * delw for w, delw in zip(self.weights, del_w)]
                self.biases = [b - (learning_rate / batch_size)
                               * delb for b, delb in zip(self.biases, del_b)]
                # update the backward matrices with weight mirroring
                self.mirror(batch_size=batch_size)

            epoch_loss = np.mean(epoch_loss)
            self.data['epoch_{}'.format(j)]['loss'] = epoch_loss

            # Log the loss
            log_str = "\nEpoch {} completed in {:.3f}s, loss: {:.3f}".format(j, time.time() - start, epoch_loss)
            self.print_and_log(log_str)

            # Evaluate on test set
            test_accuracy = self.eval(X_test, y_test)
            log_str = "Test accuracy: {}%".format(test_accuracy)
            self.print_and_log(log_str)
            self.data['epoch_{}'.format(j)]['test_accuracy'] = test_accuracy

            # Compute angles between both weights and deltas
            deltas_angles, weights_angles = self.evaluate_angles(X_train, y_train)
            self.data['epoch_{}'.format(j)]['delta_angles'] = deltas_angles
            self.data['epoch_{}'.format(j)]['weight_angles'] = weights_angles

            # save results as a json file
            with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
                json.dump(self.data, f)
