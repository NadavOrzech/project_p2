import torch
import torch.nn as nn
import os
import sys
import tqdm

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from train_results import EpochResult, FitResult


class CNNModel(nn.Module):
    def __init__(self, config, input_size=1, output_dim=2, checkpoint_file=None):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_dim = output_dim
        self.batch_size = self.config.batch_size

        if checkpoint_file is not None:
            checkpoint_dir = os.path.join('.', 'checkpoints')
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            self.checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)

        self.model1 = nn.Sequential(nn.Conv1d(in_channels=self.input_size, out_channels=400, kernel_size=3, bias=True),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        self.model2 = nn.Sequential(nn.Conv1d(in_channels=200, out_channels=100, kernel_size=2, bias=True),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    )
        self.model3 = nn.Sequential(nn.Conv1d(in_channels=50, out_channels=50, kernel_size=2, bias=True),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=4),
                                    )
        self.model4 = nn.Sequential(nn.Linear(in_features=564, out_features=100),
                                    nn.ReLU(),
                                    nn.Linear(in_features=100, out_features=self.output_dim)
                                    )

    def save_checkpoint(self, fit_result):
        torch.save(fit_result, self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the init checkpoint file
        :return: A tuple of:
                    Best test accuracy for last checkpoint
                    A EpochHeatMap object holds the attention map for all sequences for last checkpoint
                    A FitResult object containing train and test losses per epoch for last checkpoint
        """

        print(f'=== Loading checkpoint {self.checkpoint_file}, ', end='')
        data = torch.load(self.checkpoint_file, map_location=torch.device('cpu'))
        return data

    def get_dataloader(self, dataset):
        # Calculate the number of samples to include in each set.
        train_size = int((1-self.config.test_size) * len(dataset))
        test_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=self.batch_size)
        return train_dataloader, test_dataloader

    def forward(self, x):
        out = self.model1(x)
        out = self.model2(out)
        out = self.model3(out)
        out = out.view(out.size(0), -1)
        out = self.model4(out)
        return out

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader, loss_fn, optimizer):
        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # model = self.model
        # self.model.to(device)
        epochs = self.config.num_epochs
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            res_train = self.train_epoch(train_dataloader, optimizer, loss_fn, device)
            train_loss.append(res_train[0])
            train_acc.append(res_train[1])
            res_test = self.test_epoch(test_dataloader, loss_fn, device)
            test_loss.append(res_test[0])
            test_acc.append(res_test[1])
        fit_result = FitResult(epochs, train_loss, train_acc, test_loss, test_acc)
        if self.checkpoint_file is not None:
            self.save_checkpoint(fit_result)
        return fit_result

    def train_epoch(self, train_dataloader, optimizer, loss_fn, device):
        total_train_accuracy = 0
        total_train_loss = 0
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        pbar_file = sys.stdout
        pbar_name = "train_batch"
        num_batches = len(train_dataloader.batch_sampler)
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            for step, batch in enumerate(train_dataloader):
                X, y = batch[0], batch[1]
                if y.shape[0] == 1:
                    continue
                
                # Forward pass
                X = X.to(device)
                y = y.to(device)
                y_pred_log_proba = self.forward(X)
                y = torch.squeeze(y).long()  # should be of size (N,)

                # Backward pass
                optimizer.zero_grad()
                loss = loss_fn(y_pred_log_proba, y)
                loss.backward()

                # Weight updates
                optimizer.step()

                # Calculate accuracy
                total_train_loss += loss.item()
                y_pred = torch.argmax(y_pred_log_proba, dim=1)
                tp, fp, tn, fn = calculate_acc(y_pred, y)
                tp_tot += tp
                fp_tot += fp
                tn_tot += tn
                fn_tot += fn
                total_train_accuracy += torch.sum(y_pred == y).float().item()
                pbar.set_description(f'{pbar_name} ({loss.item():.3f})')
                pbar.update()
        avg_train_accuracy = (total_train_accuracy / len(train_dataloader.dataset)) * 100
        # print("  Training accuracy: {0:.2f}".format(avg_train_accuracy))
        print(f"  accuracy={avg_train_accuracy:.3f}, tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        # if tp_tot + fn_tot > 0:
        #     print(f"Pos acc: {tp_tot / (tp_tot + fn_tot):.3f},  Neg acc: {tn_tot / (tn_tot + fp_tot):.3f}")
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Log the Avg. train loss
        print("  Training loss: {0:.4f}".format(avg_train_loss))
        return EpochResult(avg_train_loss, avg_train_accuracy)

    def test_epoch(self, test_dataloader, loss_fn, device):
        total_eval_accuracy = 0
        total_eval_loss = 0
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        # Evaluate data for one epoch
        pbar_file = sys.stdout
        pbar_name = "test_batch"
        num_batches = len(test_dataloader.batch_sampler)
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            for batch in test_dataloader:
                X, y = batch[0], batch[1]
                if y.shape[0] == 1:
                    continue
                
                # Forward pass
                with torch.no_grad():
                    X = X.to(device)
                    y = y.to(device)
                    y_pred_log_proba = self.forward(X)

                    y = torch.squeeze(y).long()
                    loss = loss_fn(y_pred_log_proba, y)
                    total_eval_loss += loss.item()
                    y_pred = torch.argmax(y_pred_log_proba, dim=1)
                    tp, fp, tn, fn = calculate_acc(y_pred, y)
                    tp_tot += tp
                    fp_tot += fp
                    tn_tot += tn
                    fn_tot += fn

                total_eval_accuracy += torch.sum(y_pred == y).float().item()
                pbar.set_description(f'{pbar_name} ({loss.item():.3f})')
                pbar.update()

        avg_val_accuracy = (total_eval_accuracy / len(test_dataloader.dataset)) * 100
        print(f"  accuracy={avg_val_accuracy:.3f}, tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        # if tp_tot + fn_tot > 0:
        #     print(f"Pos acc: {tp_tot / (tp_tot + fn_tot):.3f},  Neg acc: {tn_tot / (tn_tot + fp_tot):.3f}")
        avg_val_loss = total_eval_loss / len(test_dataloader)
        # Log the Avg. validation accuracy
        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        return EpochResult(avg_val_loss, avg_val_accuracy)

def calculate_acc(y_pred, y):
    """
    Calculates the accuracy of predicted y vector
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == 0:
            if y[i] == 0:
                tn += 1
            else:
                fn += 1
        elif y[i] == 0:
            fp += 1
        else:
            tp += 1
    return tp, fp, tn, fn
