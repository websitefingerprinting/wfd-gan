from os.path import join
import torch
import torch.nn as nn
import numpy as np
import time
import random
import argparse
import configparser
import logging
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchsummary import summary
import torch.utils.data as Data
from model import DF
import utils
import common as cm

# Hyper parameters
num_epochs = 30
batch_size = 128
length = 1400
mask_length = length
DEBUG = 1


def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)
    return dict(cf['default'])


def init_logger():
    logger = logging.getLogger('df')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    formatter = logging.Formatter(LOG_FORMAT)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    random.seed(0)
    '''initialize logger'''
    logger = init_logger()
    '''read config'''
    parser = argparse.ArgumentParser(description='DF attack')
    parser.add_argument('dir',
                        metavar='<feature path>',
                        help='Path to the directory of the extracted features')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose or not')
    parser.add_argument('--cuda_id',
                        type=int,
                        default=7,
                        help='The id number of cuda to be used.')
    args = parser.parse_args()
    # Device configuration
    device = torch.device('cuda:{}'.format(args.cuda_id) if torch.cuda.is_available() else 'cpu')

    # Configure data loader
    X, y = utils.load_dataset(args.dir)
    logger.info("Loaded dataset:{}, min burst:{} max burst:{}, min label:{}, max label:{}"
                .format(X.shape, X.min(), X.max(), y.min(), y.max()))
    # reindex label starting from 0
    y -= y.min()

    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    num_classes = class_dim = len(np.unique(y))  # index start from 0

    if len(X.shape) < 3:
        X = X[:, np.newaxis, :]
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)
    seq_len = X.size(2)
    assert seq_len > 1
    assert class_dim > 1
    logger.info("X shape {}, y shape {}, class num: {}".format(X.shape, y.shape, class_dim))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)

    train_loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=2,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=2,
    )

    model = DF(length, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters())

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            # remember to deduct the class offset
            batch_y = batch_y.to(device)
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.verbose and (step + 1) % 50 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct_train = 0
            total_train = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            correct_test = 0
            total_test = 0
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_y.size(0)
                correct_test += (predicted == batch_y).sum().item()
        logger.info("Epoch [{}/{}] train acc: {:.4f} val accuracy: {:.4f}".format(
            epoch + 1, num_epochs, correct_train / total_train ,correct_test / total_test))

    fname = args.dir.split('/')[-1].split('.')[0]
    model_saved_path = join(cm.dModelDir, 'df_{}.ckpt'.format(fname))
    torch.save(model.state_dict(), model_saved_path)
    logger.info("Model is saved to {}".format(model_saved_path))
