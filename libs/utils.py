from LocAtE.libs import *
import pandas
import torch
import time
from .config import *

COLLUMN_MEAN = []
COLLUMN_STD = []


def get_auto_encoder(feature_list: list, inputs=1):
    features = feature_list.copy()
    features.insert(0, 6)
    features.append(6)
    block_block = BlockBlock(len(features) - 1, inputs, features, [1] * len(features), False, True, 1)
    return nn.Sequential(block_block,
                         nn.BatchNorm1d(6),
                         FactorizedConvModule(block_block.out_features, 6, inputs, False, 1, 0, 1, dim=1))


def get_dataset(filename):
    global COLLUMN_MEAN
    global COLLUMN_STD
    df = pandas.read_csv(filename)
    df = df.dropna()
    ndarray = df.values
    tensor = torch.DoubleTensor(ndarray).to(device)
    std, mean = torch.std_mean(tensor, 0, keepdim=True)
    tensor = (tensor - mean) / std
    COLLUMN_MEAN = mean
    COLLUMN_STD = std
    return tensor.float()


def get_item_count(tensor: torch.Tensor, log=True):
    item_count = tensor.size(0)
    if log:
        print(f"Items: {item_count}")
    return item_count


def expand_depth(tensor, target_depth):
    items = get_item_count(tensor, False) - target_depth
    return torch.stack([tensor[i:i + target_depth] for i in range(items)], dim=0).transpose(1, 2)


def print_loss(epoch, itr, pad, item_count, loss, start_time, log=True):
    if log:
        print(f"\r[{epoch}][{itr:{pad}d}/{item_count}] Loss: {loss:.4f} | Elapsed: {int(time.time() - start_time)}",
              end='')
    return None


class History:
    def __init__(self, record):
        self.data = []
        self.record = record

    def add_item(self, item):
        if self.record:
            self.data.append(item)

    def plot(self, filename, plot=True):
        if self.record:
            if plot:
                plot_hist(lwma(self.data), filename)
            print(f" | Average: {sum(self.data) / len(self.data):.4f}")
        else:
            print('')


DROPOUT = torch.ones((4, 6))
DROPOUT[0, 0] = 0
DROPOUT[1, 1] = 0
DROPOUT[2, 2] = 0
DROPOUT[2, 3] = 0
DROPOUT[3, 4] = 0
DROPOUT[3, 5] = 0


def get_dropped_out_data(data: torch.Tensor):
    random_vector = torch.randint(0, 4, (data.size(0), data.size(2)))
    return data * DROPOUT[random_vector].transpose(1, 2)


def lwma(x):
    div = ((MEAN_WINDOW ** 2 - MEAN_WINDOW) / 2)
    return [sum(x[i + j - 1] * j for j in range(1, MEAN_WINDOW + 1)) / div for i in range(len(x) - MEAN_WINDOW)]


def train_test_eval_split(tensor, test_split=0.2, eval_split=0.1):
    items = tensor.size(0)
    test_items = int(items * test_split)
    eval_items = int(items * eval_split)
    train_end = test_items + eval_items
    return tensor[:-train_end], tensor[-train_end:-eval_items], tensor[-eval_items:]


def test(model, test_data, samples=0):
    process_epoch(-1, model, test_data, test_data.size(0), train=False, plot=False, log_level=3)
    if samples:
        output = model(test_data[:samples + 1])
        output = (output * COLLUMN_STD) + COLLUMN_MEAN
        return output
    return None


def sample_print(samples):
    for s in samples:
        print(f'\t{s.tolist()}')


def evaluate(model, eval_data):
    out = test(model, eval_data, 1)
    sample_print(out)
    return out


def process_epoch(epoch, model: torch.nn.Module, expanded_trainings_data: torch.Tensor, item_count, optimizer=None,
                  log_level=2, train=True, plot=True):
    log_loss = log_level >= 2
    loss_history = History(log_level >= 3)
    start_time = time.time()
    item_count_len = len(str(item_count))
    loss = -1
    i = -1
    for i in range(0, item_count - BATCH_SIZE, BATCH_SIZE):
        target = expanded_trainings_data[i:i + BATCH_SIZE]
        source = get_dropped_out_data(target)
        model_out = model(source)
        loss = (model_out - target) ** 2
        loss = loss.mean()
        if train:
            loss.backward()
            optimizer.step()
        loss = loss.item()
        print_loss(epoch, i, item_count_len, item_count, loss, start_time, log_loss)
        loss_history.add_item(loss)
    print_loss(epoch, i, item_count_len, item_count, loss, start_time, log_level >= 1)
    loss_history.plot(f'loss_{epoch}.svg', plot=plot)
    return None


def train(model: torch.nn.Module, trainings_data: torch.Tensor, input_count, optimizer, test_data=None, epochs=-1,
          log_level=2, test_samples=1):
    trainings_data = expand_depth(trainings_data, input_count)
    if test_data is not None:
        test_data = expand_depth(test_data, input_count)
    items = get_item_count(trainings_data, log_level >= 2)
    itr = 0
    while epochs:
        process_epoch(itr, model, trainings_data, items, optimizer, log_level, train=True)
        if test_data is not None:
            out = test(model, test_data, test_samples)
            if out is not None:
                sample_print(out)
        epochs -= 1
        itr += 1
