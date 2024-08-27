import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Lambda,  Normalization, Attention, GRU, serialize, deserialize
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from dataset_combiner import combine_multiple_stocks
from header import window_size, articles_per_day, TotalStockList, max_BERT_sequence_length, num_cols, MultiTimeDistributed
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)






# Load the model, providing the custom layer
model = load_model('seer_v1.h5', custom_objects={'MultiTimeDistributed': MultiTimeDistributed, 'TFBertModel': TFBertModel})

# Get the learning rate
learning_rate = model.optimizer.learning_rate.numpy()

print('The learning rate of the model is:', learning_rate)


#Validation:
#Generate the combined dataset
validation_stocks = TotalStockList[TotalStockList.index('EIX'):(TotalStockList.index('EIX')-11)] #Grabbing stocks that we have news for, stopping at "EIX"

random.shuffle(validation_stocks)
print("Loading Datasets Now")
validation_dataset = combine_multiple_stocks(validation_stocks)

print("Batching Datasets Now")
batch_size = 12 # choose an appropriate batch size
batched_validation_dataset = validation_dataset.batch(batch_size)

# Lists to store per-batch metrics
losses = []
maes = []

# Validate the model using datasets
print("Beginning Model Validation")
for inputs, labels in batched_validation_dataset:
    results = model.test_on_batch(inputs)
    print(results)
    losses.append(results[0])
    maes.append(results[1])
print(len(maes))
print(len(losses))


print(f"Validation loss: {np.mean(losses)}, Validation MAE: {np.mean(maes)}")

# After validation, plot the per-batch loss and MAE
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Validation Loss')
plt.title('Validation Loss per Batch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(maes, label='Validation MAE')
plt.title('Validation MAE per Batch')
plt.legend()

plt.savefig('validation_history.png')
plt.show()




