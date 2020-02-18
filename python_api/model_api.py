import time

from .history import *


class ModelAPI:
    def __init__(self):
        self.intraepoch_averages = []
        self.processing_start = 0
        self.loss = 0
        self.lwma_window = 16
        self.epoch = 0
        self.working_dataset = 0
        self.training = True
        self.test_history = []
        self.train_history = []
        self.dataset = None
        self.batch_size_generator = None
        self.model = None
        self.optimizer = None
        self.print_loss = None

    def get_samples(self):
        raise UserWarning("Has to be overwritten by child classes")

    def loss_average(self):
        return sum(self.intraepoch_averages) / len(self.intraepoch_averages)

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
            self.train_history.plot('train.svg', self.lwma_window)
            self.test_history.add_item(test_loss)
            self.test_history.plot('test.svg', self.lwma_window)

            epochs -= 1
            itr += 1

    def process_dataset(self, dataset, log_level=2):
        loss_history = History(log_level >= 1, 'error')
        batch_size = self.batch_size_generator(self.epoch)
        for i in range(0, dataset.size(0) - batch_size, batch_size):
            target = dataset[i:i + batch_size]
            source = self.dataset.dropout(target)
            model_out = self.model(source)
            loss = (model_out - target).abs()
            loss = loss.mean()
            if self.training:
                loss.backward()
                self.optimizer.step()
            loss = loss.item()
            loss_history.add_item(loss)
            self.loss = loss
        return loss_history

    def process_epoch(self, dataset_list, log_level=2):
        log_loss = log_level >= 2
        for d in dataset_list:
            self.processing_start = time.time()
            self.working_dataset += 1
            loss_history = self.process_dataset(d, log_level)
            self.intraepoch_averages.append(loss_history.average())
            if log_loss:
                self.print_loss()
        self.working_dataset = 0
        return None
