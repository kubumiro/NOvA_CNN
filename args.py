import argparse

# Filepaths are relative to the Wilson Cluster
default_args = { "traindir" : "/mnt/lustre/helios-home/kubumiro/dl/TensorFlow-GPU/Masters/Masters/Data_split/data_train",
                 "validdir" : "/mnt/lustre/helios-home/kubumiro/dl/TensorFlow-GPU/Masters/Masters/Data_split/data_valid",
                 "testdir" : "/mnt/lustre/helios-home/kubumiro/dl/TensorFlow-GPU/Masters/Masters/Data_split/data_test",
                 "nepochs" : 100,
                 "ntrsteps" : 3500,
                 "ntssteps" : 3500,
                 "trbatchsize" : 16,
                 "tsbatchsize" : 16,
                 "momentum" : 0.9,
                 "weightdecay" : 0.0002,
                 "baselr" : 0.002,
                 "reducecosmics" : 1
                }

def parse_args():
    parser = argparse.ArgumentParser(
                   description='Arguments for CVN Short Simple Keras training.')
    parser.add_argument(
                '--traindir',
                help='The path to the directory of HDF5 training files.',
                nargs='?',
                default=default_args['traindir'],
                type=str)
    parser.add_argument(
                '--testdir',
                help='The path to the directory of HDF5 test files.',
                nargs='?',
                default=default_args['testdir'],
                type=str)
    parser.add_argument(
                '--validdir',
                help='The path to the directory of HDF5 test files.',
                nargs='?',
                default=default_args['validdir'],
                type=str)
    parser.add_argument(
                  '--nepochs',
                  help='Intervals between validation steps.',
                  nargs='?',
                  default=default_args['nepochs'],
                  type=int)
    parser.add_argument(
                  '--ntrsteps',
                  help='Number of steps per training interval.',
                  nargs='?',
                  default=default_args['ntrsteps'],
                  type=int)
    parser.add_argument(
                  '--ntssteps',
                  help='Number of testing iterations per validation.',
                  nargs='?',
                  default=default_args['ntssteps'],
                  type=int)
    parser.add_argument(
                  '--trbatchsize',
                  help='Training batch size.',
                  nargs='?',
                  default=default_args['trbatchsize'],
                  type=int)
    parser.add_argument(
                  '--tsbatchsize',
                  help='Testing batch size.',
                  nargs='?',
                  default=default_args['tsbatchsize'],
                  type=int)
    parser.add_argument(
                  '--momentum',
                  help='The momentum parameter for SGD.',
                  nargs='?',
                  default=default_args['momentum'],
                  type=float)
    parser.add_argument(
                  '--weightdecay',
                  help='The weight decay multiplier for L2 regularization.',
                  nargs='?',
                  default=default_args['momentum'],
                  type=float)
    parser.add_argument(
                  '--baselr',
                  help='The base learning rate',
                  nargs='?',
                  default=default_args['baselr'],
                  type=float)
    parser.add_argument('-l', '--label', action='store_true',
                        help='Set this flag to preprocess the dataset from CAF.')
    parser.add_argument('--reducecosmics', action='store_true',
                        help='Set this flag to reduce cosmics to 10 percent.')
    args = parser.parse_args()

    return args
