from LocAtE.libs import *
from libs import *

model = get_auto_encoder(FEATURE_LIST, INPUTS)
dataset = get_dataset('nextbike.csv')
train_data, test_data, eval_data = train_test_eval_split(dataset)
model, optimizer = get_model(model, LEARNING_RATE, device)
print(model)
print(f'Parameters: {parameter_count(model)}')
train(model, train_data, INPUTS, optimizer, test_data=test_data, log_level=1)
evaluate(model, eval_data)
