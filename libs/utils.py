from LocAtE.libs import *
import pandas
import torch
import time
import os
from .config import *


def lwma(x):
    div = ((MEAN_WINDOW ** 2 - MEAN_WINDOW) / 2)
    return [sum(x[i + j - 1] * j for j in range(1, MEAN_WINDOW + 1)) / div for i in range(len(x) - MEAN_WINDOW)]


class Dataset:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.dataset = []
        self.test_dataset = []
        self.eval_dataset = []
        self.dropout_mask = torch.ones((1, 6))

    def get_dropout_mask(self, *args):
        item_counts = [len(a) for a in args]
        arg_count = len(args)
        total_item_count = prod(item_counts)
        dropout = torch.ones((total_item_count, 6)).to(device)
        for i in range(total_item_count):
            items = [args[idx][i % item_counts[idx]] for idx in range(arg_count)]
            items = list(flatten(items))
            dropout[i, items] = 0
        self.dropout_mask = dropout
        return None

    def dropout(self, item):
        random_vector = torch.randint(0, self.dropout_mask.size(0), (item.size(0), item.size(2)))
        return self.dropout_mask[random_vector].transpose(1, 2) * item

    def items(self):
        return sum([i.size(0) for i in self.dataset])

    def split(self, test_split=0.2, eval_split=0.1):
        for i, d in enumerate(self.dataset):
            items = d.size(0)
            test_items = int(items * test_split)
            eval_items = int(items * eval_split)
            train_items = items - test_items - eval_items
            self.test_dataset.append(d[train_items:-eval_items])
            self.eval_dataset.append(d[-eval_items:])
            self.dataset[i] = d[:train_items]

    def expand(self, target_depth):
        for idx in range(len(self.dataset)):
            items = self.dataset[idx].size(0) - target_depth
            self.dataset[idx] = torch.stack([self.dataset[idx][i:i + target_depth] for i in range(items)],
                                            dim=0).transpose(1, 2)

    def add(self, filename):
        df = pandas.read_csv(filename)
        df = df.dropna()
        ndarray = df.values

        tensor = torch.DoubleTensor(ndarray).to(device)
        std, mean = torch.std_mean(tensor, 0, keepdim=True)
        tensor = (tensor - mean) / std

        prev_items = self.items()
        new_items = tensor.size(0)

        self.mean = (self.mean * prev_items + mean * new_items) / (prev_items + new_items)
        self.std = (self.std * prev_items + std * new_items) / (prev_items + new_items)

        tensor = tensor.float()

        self.dataset.append(tensor)
        return None


class History:
    def __init__(self, record, plot_folder):
        self.data = []
        self.record = record
        self.plot_folder = plot_folder

    def add_item(self, item):
        self.data.append(item)

    def average(self):
        return sum(self.data) / len(self.data)

    def lwma(self):
        return lwma(self.data)

    def plot(self, filename):
        try:
            os.mkdir(self.plot_folder)
        except OSError:
            pass
        plot_hist(self.data, f'{self.plot_folder}/raw_{filename}')
        plot_hist(self.lwma(), f'{self.plot_folder}/lwma_{filename}')
        return None


class AutoEncoder:
    def __init__(self, feature_list: list, inputs=1):
        features = feature_list.copy()
        features.insert(0, 6)
        features.append(6)
        block_block = BlockBlock(len(features) - 1, inputs, features, [1] * len(features), False, True, 1)

        self.model = nn.Sequential(block_block, nn.BatchNorm1d(6),
                                   FactorizedConvModule(block_block.out_features, 6, inputs, False, 1, 0, 1, dim=1))
        self.model, self.optimizer = get_model(self.model, LEARNING_RATE, device)
        self.dataset = Dataset()

        self.train_history = History(True, 'plots')
        self.test_history = History(True, 'plots')
        self.inputs = inputs

        self.training = True
        self.epoch = 0
        self.loss = 0
        self.processing_start = 0
        self.working_dataset = 0
        self.samples = []
        self.intraepoch_averages = []

    def print_loss(self):
        print(
                f"\r[{self.epoch}][{self.working_dataset}/{len(self.dataset.dataset)}] Loss: {self.loss:.4f} | Elapsed: {int(time.time() - self.processing_start)}",
                end='')
        return None

    def print_parameters(self):
        print(f"Parameters: {parameter_count(self.model)}")

    def print_samples(self):
        for s in self.samples:
            print(f'\t{s.tolist()}')

    def loss_average(self):
        return sum(self.intraepoch_averages) / len(self.intraepoch_averages)

    def get_samples(self, samples, print_samples=False):
        if samples:
            output = self.model(self.dataset.test_dataset[0][:samples + 1])
            output = (output * self.dataset.std) + self.dataset.mean
            if print_samples:
                self.print_samples()
            return output
        else:
            return None

    def add_datasets(self, *datasets):
        for d in datasets:
            self.dataset.add(d)
        self.dataset.expand(self.inputs)
        return None

    def add_mask(self, *args):
        return self.dataset.get_dropout_mask(*args)

    def test(self):
        return self._processing_wrapper(False, dataset_list=self.dataset.test_dataset, log_level=1)

    def evaluate(self):
        return self._processing_wrapper(False, dataset_list=self.dataset.eval_dataset, log_level=1)

    def _processing_wrapper(self, training, **kwargs):
        self.training = training
        self.process_epoch(**kwargs)
        return self.loss_average()

    def train(self, epochs, samples=0, log_level=1):
        itr = 0
        while epochs:
            self.epoch = itr
            self.training = True
            self.process_epoch(self.dataset.dataset, log_level=log_level)
            train_loss = self._processing_wrapper(True, dataset_list=self.dataset.dataset, log_level=log_level)
            test_loss = self.test()

            self.get_samples(samples, True)

            self.train_history.add_item(train_loss)
            self.train_history.plot('train.svg')
            self.test_history.add_item(test_loss)
            self.test_history.plot('test.svg')

            epochs -= 1
            itr += 1

    def process_dataset(self, dataset, log_level=2):
        loss_history = History(log_level >= 1, 'error')
        for i in range(0, dataset.size(0) - BATCH_SIZE, BATCH_SIZE):
            target = dataset[i:i + BATCH_SIZE]
            source = self.dataset.dropout(target)
            model_out = self.model(source)
            loss = (model_out - target).abs()
            loss = loss.mean()
            if self.training:
                loss.backward()
                self.optimizer.step()
            self.loss = loss.item()
            loss_history.add_item(loss)
        return loss_history

    def process_epoch(self, dataset_list, log_level=2):
        log_loss = log_level >= 2
        self.intraepoch_averages = []
        for d in dataset_list:
            self.processing_start = time.time()
            self.working_dataset += 1
            loss_history = self.process_dataset(d, log_level)
            self.intraepoch_averages.append(loss_history.average())
            if log_loss:
                self.print_loss()
        self.working_dataset = 0
        return None

    def __str__(self):
        return str(self.model)
