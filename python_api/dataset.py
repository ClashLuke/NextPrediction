import pandas

from LocAtE.libs import *


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
