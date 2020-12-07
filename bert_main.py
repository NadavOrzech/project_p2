import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from config import Config
from BertClassifier import BertClassifier, create_data_list
from BertEmbedder import BertEmbedder
from torch.utils.data import TensorDataset
from rnn import LSTMModel
from cnn import CNNModel
from plot import plot_fit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, ParameterGrid, train_test_split
# from training import get_dataloader
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from train_results import FitResult

BERT_CLASSIFIER = "bert_classifier"
RNN_CLASSIFIER = "RNN_classifier"
CNN_CLASSIFIER = "CNN_classifier"


def plot_graphs(classifier_name):
    fit_result = None
    checkpoint_dir = os.path.join('.', 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, classifier_name)
    if os.path.isfile(
            checkpoint_file):  # Loading the checkpoints if the models already trained with the same hyperparameters
        fit_result = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        plot_fit(fit_result, 'RNN classifier graph', legend='total')


def run_bert_classifier(config):
    headlines_list, labels = create_data_list(config.input_path)
    bert_classifier = BertClassifier(headlines_list, labels, config, BERT_CLASSIFIER)
    dataset = bert_classifier.get_dataset()
    train_dataloader, test_dataloader = bert_classifier.get_dataloader(dataset)
    fit_result = bert_classifier.fit(train_dataloader, test_dataloader)


def run_LSTM(config):
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    embeddings, labels = bert_embedder.get_word_embeddings()

    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=config.test_size)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=config.batch_size)
    # dataset = TensorDataset(embeddings, labels)
    model = LSTMModel(config, checkpoint_file=RNN_CLASSIFIER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # train_dataloader, test_dataloader = get_dataloader(dataset, config.test_size, config.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    fit_result = model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)


def run_cnn(config):
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    embeddings, labels = bert_embedder.get_sentence_embeddings()
    embeddings = embeddings.unsqueeze(0).permute(1, 0, 2)

    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=config.test_size)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=config.batch_size)
    # dataset = TensorDataset(embeddings, labels)
    model = CNNModel(config, checkpoint_file=CNN_CLASSIFIER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # train_dataloader, test_dataloader = get_dataloader(dataset, config.test_size, config.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    fit_result = model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)


def rnn_cross_validation_model(config, n_splits):
    scores = {}
    kf = KFold(n_splits=n_splits)
    params_grid = {'l_r': [3e-5, 5e-5, 1e-4], 'batch_size': [32, 64], 'dropout': [0.2, 0.4, 0.6],
                   'hidden_dim': [200, 400]}  # todo: needs to add wanted params
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    print("started embedding")
    embeddings, labels = bert_embedder.get_word_embeddings()
    print("finished embedding")
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=config.test_size)
    max_acc = -numpy.inf

    for index, params in enumerate(list(ParameterGrid(param_grid=params_grid))):
        config.dropout = params['dropout']
        config.hidden_dim = params['hidden_dim']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("-----------------------------------------------------------------")
        print(f"current params are {params}")
        print(f" index is: {index}")

        curr_acc = 0
        for train_idx, valid_idx in kf.split(x_train):
            lstm = LSTMModel(config, checkpoint_file=None)
            lstm.to(device)
            lstm.batch_size = params['batch_size']
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(lstm.parameters(), lr=params["l_r"])

            train_x, train_y = x_train[train_idx], y_train[train_idx]
            train_set = TensorDataset(train_x, train_y)
            valid_x, valid_y = x_train[valid_idx], y_train[valid_idx]
            valid_set = TensorDataset(valid_x, valid_y)
            train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=lstm.batch_size)
            valid_dataloader = DataLoader(valid_set, sampler=SequentialSampler(valid_set), batch_size=lstm.batch_size)
            fit_result = lstm.fit(train_dataloader, valid_dataloader, loss_fn, optimizer)
            curr_acc += max(fit_result.test_acc)
        mean = curr_acc / n_splits
        if mean > max_acc:
            max_acc = mean
            best_params = params
        scores[index] = {
            "params": params,
            "acc": mean
        }

    print(f"all scores: {scores}")
    print(f"max accuracy achieved is {max_acc}")
    print(f"best params are {best_params}")

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=best_params['batch_size'])
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=best_params['batch_size'])
    best_model = LSTMModel(config, checkpoint_file=RNN_CLASSIFIER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)
    best_model.batch_size = best_params['batch_size']
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["l_r"])
    fit_result = best_model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)


def cnn_cross_validation_model(config, n_splits):
    scores = {}
    kf = KFold(n_splits=n_splits)
    params_grid = {'l_r': [3e-5, 5e-5, 1e-4, 1e-3], 'batch_size': [32, 64]}
    headlines_list, labels = create_data_list(config.input_path)
    bert_embedder = BertEmbedder(headlines_list, labels, config)
    print("started embedding")
    embeddings, labels = bert_embedder.get_sentence_embeddings()
    print("finished embedding")
    embeddings = embeddings.unsqueeze(0).permute(1, 0, 2)
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=config.test_size)
    max_acc = -numpy.inf

    for index, params in enumerate(list(ParameterGrid(param_grid=params_grid))):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("-----------------------------------------------------------------")
        print(f"current params are {params}")
        print(f" index is: {index}")

        curr_acc = 0
        for train_idx, valid_idx in kf.split(x_train):
            cnn = CNNModel(config, checkpoint_file=CNN_CLASSIFIER)
            cnn.to(device)
            cnn.batch_size = params['batch_size']
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(cnn.parameters(), lr=params["l_r"])

            train_x, train_y = x_train[train_idx], y_train[train_idx]
            train_set = TensorDataset(train_x, train_y)
            valid_x, valid_y = x_train[valid_idx], y_train[valid_idx]
            valid_set = TensorDataset(valid_x, valid_y)
            train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=cnn.batch_size)
            valid_dataloader = DataLoader(valid_set, sampler=SequentialSampler(valid_set), batch_size=cnn.batch_size)
            fit_result = cnn.fit(train_dataloader, valid_dataloader, loss_fn, optimizer)
            curr_acc += max(fit_result.test_acc)
        mean = curr_acc / n_splits
        scores[f"{params}"] = mean
        if mean > max_acc:
            max_acc = mean
            best_params = params
        scores[index] = {
            "params": params,
            "acc": mean
        }
    print(f"all scores: {scores}")
    print(f"max accuracy achieved is {max_acc}")
    print(f"best params are {best_params}")

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=best_params['batch_size'])
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=best_params['batch_size'])
    best_model = CNNModel(config, checkpoint_file=CNN_CLASSIFIER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)
    best_model.batch_size = best_params['batch_size']
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["l_r"])
    fit_result = best_model.fit(train_dataloader, test_dataloader, loss_fn, optimizer)


if __name__ == "__main__":
    config = Config()
    # plot_graphs(RNN_CLASSIFIER)  # needs to  insert classifier name
    # rnn_cross_validation_model(config, 4)
    run_bert_classifier(config)
    # run_cnn(config)
    # run_LSTM(config)


