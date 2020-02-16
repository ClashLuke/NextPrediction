from LocAtE.libs import *
from libs import *




model = get_auto_encoder(FEATURE_LIST, INPUTS)
print(model)
trainings_data = get_trainings_data('nextbike.csv')
model, optimizer = get_model(model, LEARNING_RATE, device)
train(model, trainings_data, INPUTS, optimizer, log_level=3)
