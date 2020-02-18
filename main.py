from python_api import AutoEncoder

inputs = 1
feature_list = [6 * 32] * 2

model = AutoEncoder(feature_list, inputs)

print(model)
print(f'Parameters: {model.parameters}')

model.add_datasets('nextbike.csv')
model.dataset.split()
model.add_mask([0, 1], [[2, 3], [4, 5]])

model.add_batch_size_schedule(4096)
model.train(-1, 1)
model.evaluate()
