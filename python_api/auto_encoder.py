from .dataset import *
from .model_api import *
from .history import *


class AutoEncoder(ModelAPI):
    def __init__(self, feature_list: list, inputs=1, learning_rate=1e-2):
        super().__init__()
        features = feature_list.copy()
        features.insert(0, 6)
        features.append(6)

        self.model = BlockBlock(len(features) - 1, inputs, features, [1] * len(features), False, True, 1)
        self.model, self.optimizer = get_model(self.model, learning_rate, device)
        self.parameters = parameter_count(self.model)
        self.dataset = Dataset()

        self.train_history = History(True, 'plots')
        self.test_history = History(True, 'plots')
        self.inputs = inputs

        self.samples = []
        self.batch_size_generator = lambda x: None

    def print_loss(self):
        print(f"[{self.epoch}][{self.working_dataset}/{len(self.dataset.dataset)}] Loss: {self.loss:.4f} | Elapsed: {int(time.time() - self.processing_start)}")
        return None

    def print_samples(self):
        for s in self.samples:
            print(f'\t{s.tolist()}')

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

    def add_batch_size_schedule(self, batch_size=None):
        if isinstance(batch_size, type(None)):
            raise UserWarning("No batch size given.")

        if isinstance(batch_size, int):
            self.batch_size_generator = lambda x: batch_size
        elif isinstance(batch_size, type(self.__init__)):
            self.batch_size_generator = lambda x: batch_size(x)
        elif isinstance(batch_size, list):
            l = len(batch_size) - 1
            self.batch_size_generator = lambda x: batch_size[max(x, l)]
        else:
            raise UserWarning(f"Unknown type {type(batch_size)} for batch size. Please use int, list or function.")

    def __str__(self):
        return str(self.model)
