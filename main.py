from libs import *

model = AutoEncoder(FEATURE_LIST, INPUTS)

print(model)
print(f'Parameters: {model.parameters}')

model.add_datasets('nextbike.csv')
model.dataset.split()
model.add_mask([0, 1], [[2, 3], [4, 5]])

model.train(-1, 1, 1)
model.evaluate()
