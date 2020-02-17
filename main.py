from LocAtE.libs import *
from libs import *

model = AutoEncoder(FEATURE_LIST, INPUTS)

model.print_parameters()
print(model)

model.add_datasets('nextbike.csv')
model.dataset.split()

model.train(-1, 1, 1)
model.evaluate()
