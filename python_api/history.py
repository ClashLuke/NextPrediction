import os

from LocAtE.libs import plot_hist


class History:
    def __init__(self, record, plot_folder):
        self.data = []
        self.record = record
        self.plot_folder = plot_folder

    def add_item(self, item):
        self.data.append(item)

    def average(self):
        return sum(self.data) / len(self.data)

    def lwma(self, mean_window):
        div = ((mean_window ** 2 + mean_window) / 2)
        return [sum(self.data[i + j - 1] * j for j in range(1, mean_window + 1)) / div
                for i in range(len(self.data) - mean_window)]

    def plot(self, filename, mean_window):
        try:
            os.mkdir(self.plot_folder)
        except OSError:
            pass
        plot_hist(self.data, f'{self.plot_folder}/raw_{filename}')
        plot_hist(self.lwma(mean_window), f'{self.plot_folder}/lwma_{filename}')
