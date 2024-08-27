import tensorflow as tf
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Lambda,  Normalization, Attention, GRU, serialize, deserialize, MultiHeadAttention, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K, regularizers
from dataset_combiner import combine_multiple_stocks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adagrad #try Backtracking Gradient descent
from header import window_size, articles_per_day, TotalTrainList, max_BERT_sequence_length, num_cols, MultiTimeDistributed, TotalTestList, ScalingLayer, custom_loss, custom_binary_loss
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras import mixed_precision
import gc

# Set to True for training on a single stock, set to False for batch of stocks
single_stock_mode = False  # or False

# Define the stock to be used in single stock mode
single_stock_name = 'FFIV'#TotalTrainList[0]  # Replace with the actual stock name



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)






# Load the BERT model from huggingface
bert_model = TFBertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", from_pt=True)
bert_model.get_layer("bert").pooler.trainable = False

# Define two inputs
stock_data_input = Input(shape=(window_size, num_cols), name="Stock Data")  # Input for the stock market data
#news_article_input = Input(shape=(window_size, articles_per_day, max_BERT_sequence_length), name="News Data", dtype=tf.int32)  # Input for the tokenized text data

# Define a layer to average embeddings
#average_layer = Lambda(lambda x: tf.reduce_mean(x, axis=1))

# Define the second branch: a time-distributed BERT that applies BERT to all articles per day
#bert_branch = MultiTimeDistributed(bert_model)(news_article_input)
# Use the pooled output (CLS token)
#bert_branch = Lambda(lambda x: x[:, :, :, 0])(bert_branch)

# Normalize the averaged BERT output
#bert_branch = Normalization(axis=1)(bert_branch)

# Combine the output of the two branches
#x = concatenate([stock_data_input, bert_branch])
item_attention = Attention()([stock_data_input, stock_data_input])

x = item_attention

# First Transformer block
'''multi_head_attention1 = MultiHeadAttention(num_heads=6, key_dim=30, dropout=0.0)
x = multi_head_attention1(query=x, value=x)  # The output of the transformer layer is used

multi_head_attention2 = MultiHeadAttention(num_heads=6, key_dim=30, dropout=0.0)
x = multi_head_attention2(query=x, value=x)  # The output of the transformer layer is used

multi_head_attention3 = MultiHeadAttention(num_heads=6, key_dim=30, dropout=0.0)
x = multi_head_attention3(query=x, value=x)  # The output of the transformer layer is used

multi_head_attention4 = MultiHeadAttention(num_heads=6, key_dim=30, dropout=0.0)
x = multi_head_attention4(query=x, value=x)  # The output of the transformer layer is used

multi_head_attention5 = MultiHeadAttention(num_heads=6, key_dim=30, dropout=0.0)
x = multi_head_attention5(query=x, value=x)  # The output of the transformer layer is used
'''
#x = BatchNormalization(axis=2)(x)

# Pass output of transformer/attention layers into LSTM block
x = LSTM(96*3, return_sequences=True)(x)
x = LSTM(96*2, return_sequences=True)(x)
seq_attention = Attention()([x, x]) 
x = LSTM(96*1)(seq_attention)

# Pass output of LSTM into dense layers for output
x = Dense(96, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(112, activation='relu')(x)
x = Dense(112, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(112, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(112, activation='relu')(x)
output = Dense(1, activation='tanh')(x)

#x = Dense(1, activation='tanh')(x)
#output = ScalingLayer(50)(x)



# Create and compile the model
#model = Model(inputs=[news_article_input, stock_data_input], outputs=output)
model = Model(inputs=[stock_data_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0007), loss=custom_binary_loss, metrics=['MeanAbsoluteError'])
#model.compile(optimizer=Adam(learning_rate=0.000001), loss=custom_loss, metrics=['MeanAbsoluteError'])
print(model.summary())



# Plot and save the model architecture
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Load the model weights if they exist
weights_path = 'seer_v1_weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)

# Define a function to divide stock_names into chunks of size 15
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Generate the combined dataset
# Check if single_stock_mode is enabled
if single_stock_mode:
    # In single_stock_mode, use only the defined single stock
    stock_names = [single_stock_name]
else:
    # Otherwise, use the batch of stocks
    stock_names = TotalTrainList

random.shuffle(stock_names)
#random.SystemRandom().shuffle(stock_names)
# If single_stock_mode, no need to chunk
if single_stock_mode:
    stock_name_chunks = [stock_names]
else:
    stock_name_chunks = list(chunks(stock_names, len(TotalTrainList)))  # Divide stock_names into chunks of size 15

#early_stopping = EarlyStopping(monitor='loss', patience=5)

SAVED_CHUNK_PLACE = 1

for i, stock_name_chunk in enumerate(stock_name_chunks):
    if i+1 < SAVED_CHUNK_PLACE:
        continue
    print(f"Loading Datasets Now for Chunk {i+1}/{len(stock_name_chunks)}")
    combined_dataset = combine_multiple_stocks(stock_name_chunk)

    print("Batching Datasets Now")
    batch_size = 1024#6 #12 # choose an appropriate batch size
    batched_train_dataset = combined_dataset.batch(batch_size)

    # Validation step
    # Select 5 random stocks for validation
    random.shuffle(TotalTestList)
    validation_stocks = TotalTestList

    print("Loading Validation Datasets Now")
    validation_dataset = combine_multiple_stocks(validation_stocks)

    print("Batching Validation Datasets Now")
    batched_validation_dataset = validation_dataset.batch(batch_size)

    # Train the model using datasets
    print(f"Beginning Model Training for Chunk {i+1}/{len(stock_name_chunks)}")
    history = model.fit(batched_train_dataset, validation_data=batched_validation_dataset, epochs=5)
    #history = model.fit(batched_train_dataset, epochs=15)

    # Save the weights after each chunk
    model.save_weights(weights_path)

    # Save the complete model
    model.save("seer_v1.h5")

    # Clear session
    tf.keras.backend.clear_session()

    # Explicitly delete datasets and call garbage collector
    del combined_dataset, batched_train_dataset#, validation_dataset, batched_validation_dataset
    gc.collect()

# Print model summary
model.summary()

# After training, plot the training history
plt.figure(figsize=(12, 4))
plt.plot(history.history['mean_absolute_error'], label='Train Growth Percent Error')
plt.plot(history.history['val_mean_absolute_error'], label='Val Growth Percent Error')  # Add validation error to plot
plt.title('MAE evolution')
plt.legend()
plt.savefig('training_history.png')
plt.show()


