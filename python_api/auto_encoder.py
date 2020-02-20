import time

from .LocAtE.libs import BlockBlock, DEVICE, get_model, parameter_count
from .dataset import Dataset
from .model_api import ModelAPI


class AutoEncoder(ModelAPI):
    def __init__(self, feature_list: list, inputs=1, learning_rate=1e-2):
        super().__init__()
        features = feature_list.copy()
        features.insert(0, 6)
        features.append(6)

        self.model = BlockBlock(len(features) - 1, inputs, features,
                                [1] * len(features), False, True, 1)
        self.model, self.optimizer = get_model(self.model, learning_rate, DEVICE)
        self.parameters = parameter_count(self.model)
        self.dataset = Dataset()

        self.inputs = inputs

        self.sample_list = []
        self.batch_size_generator = lambda x: None

    def print_loss(self):
        print(
                f"[{self.epoch}][{self.working_dataset}/{len(self.dataset.dataset)}] "
                f"Loss: {self.loss:.4f} | Elapsed:"
                f" {int(time.time() - self.processing_start)}")

    def print_samples(self):
        for sample in self.sample_list:
            print(f'\t{sample.tolist()}')

    def get_samples(self, samples, print_samples):
        output = self.model(self.dataset.test_dataset[0][:samples + 1])
        self.sample_list = (output * self.dataset.std) + self.dataset.mean
        if print_samples:
            self.print_samples()
        return output

    def add_datasets(self, *datasets):
        for file_name in datasets:
            self.dataset.add(file_name)
        self.dataset.expand(self.inputs)

    def add_mask(self, *args):
        return self.dataset.get_dropout_mask(*args)

    def add_batch_size_schedule(self, batch_size=None):
        if isinstance(batch_size, type(None)):
            raise UserWarning("No batch size given.")

        if isinstance(batch_size, int):
            self.batch_size_generator = lambda x: batch_size
        elif isinstance(batch_size, type(self.__init__)):
            self.batch_size_generator = batch_size
        elif isinstance(batch_size, list):
            batch_size_elements = len(batch_size) - 1
            self.batch_size_generator = lambda x: batch_size[
                min(x, batch_size_elements)]
        else:
            raise UserWarning(f"Type {type(batch_size)} unsupported. Use list, int or "
                              f"function")

    def __str__(self):
        return str(self.model)
