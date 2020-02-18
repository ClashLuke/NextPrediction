import time

from .history import History


class ModelAPI:
    def __init__(self):
        self.intraepoch_averages = []
        self.processing_start = 0
        self.loss = 0
        self.lwma_window = 16
        self.epoch = 0
        self.working_dataset = 0
        self.log_level = 2
        self.training = True
        self.train_history = History(True, 'plots')
        self.test_history = History(True, 'plots')
        self.samples = True
        self.dataset = None
        self.batch_size_generator = lambda x: 128 * (x + 1)
        self.model = lambda x: x
        self.optimizer = None

    @staticmethod
    def get_samples(*args, **kwargs):
        raise NotImplementedError("Has to be overwritten by child classes")

    @staticmethod
    def print_loss():
        raise NotImplementedError("Has to be overwritten by child classes")

    def loss_average(self):
        return sum(self.intraepoch_averages) / len(self.intraepoch_averages)

    def test(self):
        return self._processing_wrapper(False, dataset_list=self.dataset.test_dataset)

    def evaluate(self):
        return self._processing_wrapper(False, dataset_list=self.dataset.eval_dataset)

    def _processing_wrapper(self, training, **kwargs):
        self.training = training
        log_level = self.log_level
        if not training:
            self.log_level = 1
        self.process_epoch(**kwargs)
        if self.log_level >= 1:
            self.log_level = log_level
            return self.loss_average()
        return -1

    def train(self, epochs, samples=0):
        itr = 0
        while epochs:
            self.epoch = itr
            self.training = True
            self.process_epoch(self.dataset.dataset)
            train_loss = self._processing_wrapper(True,
                                                  dataset_list=self.dataset.dataset)
            test_loss = self.test()

            if self.samples:
                self.get_samples(samples, True)

            self.train_history.add_item(train_loss)
            self.train_history.plot('train.svg', self.lwma_window)
            self.test_history.add_item(test_loss)
            self.test_history.plot('test.svg', self.lwma_window)

            epochs -= 1
            itr += 1

    def process_dataset(self, dataset):
        loss_history = History(self.log_level >= 1, 'error')
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

    def process_epoch(self, dataset_list):
        log_loss = self.log_level >= 2
        for dataset in dataset_list:
            self.processing_start = time.time()
            self.working_dataset += 1
            loss_history = self.process_dataset(dataset)
            self.intraepoch_averages.append(loss_history.average())
            if log_loss:
                self.print_loss()
        self.working_dataset = 0
