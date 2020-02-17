from libs import *

model = AutoEncoder(FEATURE_LIST, INPUTS)

model.print_parameters()
print(model)

model.add_datasets('nextbike.csv')
model.dataset.split()
model.add_mask([0, 1], [[2, 3], [4, 5]])

model.train(-1, 1, 1)
model.evaluate()
