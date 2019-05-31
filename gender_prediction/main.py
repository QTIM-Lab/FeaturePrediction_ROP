import sys

from utils import *
from prepare import *
from model import *
from train import *
from evaluate import *

if __name__ == "__main__":
    action = sys.argv[1] # prepare, train, eval, predict
    data_dir = sys.argv[2] # file of train, test data
    csv_file = sys.argv[3] # csv file of info OR output file name
    model_path = sys.argv[4] # model path save or load OR dataloader
    # TODO: batch_size, num_epochs

    if action == 'prepare':
        prepare(action, data_dir, csv_file, model_path)
    elif action == 'train':
        dataloader, data_size = prepare(action, data_dir, csv_file)
        train(model_path, dataloader, data_size)
    elif action == 'eval':
        dataloader, data_size = prepare(action, data_dir, csv_file)
        evaluate(model_path, dataloader, data_size)
    elif action == 'predict':
        # TODO: dataloader  = prepare(action, data_dir)
        predict(model_path, dataloader)