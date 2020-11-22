from fcnn.FCNN_WM import FCNN_WM
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')


def parse_args():
    '''
    Parse arguments from command line input
    '''
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--n_epochs', type=int, default='400', help="The number of epochs")
    parser.add_argument('--size_hidden_layers', type=int, nargs='+', default=[1000, 200], help="The number of hidden neurons per layer")
    parser.add_argument('--batch_size', type=int, default='128', help="The training batch size")
    parser.add_argument('--learning_rate', type=float, default='0.2', help="The training batch size")
    parser.add_argument('--test_frequency', type=int, default='1', help="The number of epochs after which the model is tested")
    parser.add_argument('--save_dir', type=str, default='./experiments', help="The folder path to save the experimental config, logs, model")
    parser.add_argument('--seed', type=int, default=1111, help='random seed for Numpy')
    args, unknown = parser.parse_known_args()
    return args


def preprocess(normalize=True):
    '''
    Description: helper function to load and preprocess the dataset
    Params: normalize = a boolean to specify if the dataset should be normalized
    Outputs: Pre-processed image features and labels
    '''
    
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    # Normalization of pixel values to [0-1] range
    if normalize:
        X_train /= 255
        X_test /= 255
    return (X_train, Y_train), (X_test, Y_test)


def main():
    # Parse arguments
    args = parse_args()

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)

    # Use flags passed to the script to make the name for the experimental dir
    experiment_path = os.path.join(args.save_dir, "mnist")

    print('\n########## Setting Up Experiment ######################')
    # Increment a counter so that previous results with the same args will not be overwritten.
    i = 0
    while os.path.exists(experiment_path + "-V" + str(i)):
        i += 1
    experiment_path = experiment_path + "-V" + str(i)

    # Creates an experimental directory and dumps all the args to a text file
    os.makedirs(experiment_path)
    print("\nPutting log in {}".format(experiment_path))
    with open(os.path.join(experiment_path, 'experiment_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(arg + '    ' + str(value) + '\n')

    # load and preprocess the dataset
    (X_train, Y_train), (X_test, Y_test) = preprocess()

    # Add the size of the input and output layer depending on the dataset
    size_layers = [X_train.shape[1]] + args.size_hidden_layers + [Y_train.shape[1]]

    log_str = ("\n" + "=" * 20 + "\n") + \
        "\tbatch_size: {}\n".format(args.batch_size) + \
        "\tlearning rate: {}\n".format(args.learning_rate) + \
        "\tn_epochs: {}\n".format(args.n_epochs) + \
        "\ttest frequency: {}\n".format(args.test_frequency) + \
        "\tsize_layers: {}\n".format(size_layers) + \
        "=" * 20 + "\n"
    print(log_str)
    with open(os.path.join(experiment_path, 'log.txt'), 'a') as f_:
        f_.write(log_str + '\n')

    model = FCNN_WM(size_layers, experiment_path)

    # Run the training
    model.train(X_train, Y_train, X_test, Y_test, args.batch_size, args.learning_rate, args.n_epochs, args.test_frequency)


if __name__ == '__main__':
    main()
