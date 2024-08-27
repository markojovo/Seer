import tensorflow as tf
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Lambda, Normalization, Attention, GRU, serialize, deserialize
from tensorflow.keras import backend as K
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import random
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model
from header import window_size, articles_per_day, TotalStockList, max_BERT_sequence_length, num_cols, MultiTimeDistributed
from dataset_combiner import combine_multiple_stocks

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)



# Define a function for the model building which takes hyperparameters as arguments
def build_model(hp: HyperParameters):
    # Load the BERT model from huggingface
    bert_model = TFBertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", from_pt=True)
    bert_model.get_layer("bert").pooler.trainable = False

    # Define two inputs
    stock_data_input = Input(shape=(window_size, num_cols), name="Stock Data")  # Input for the stock market data
    news_article_input = Input(shape=(window_size, articles_per_day, max_BERT_sequence_length), name="News Data", dtype=tf.int32)  # Input for the tokenized text data

    # Define a layer to average embeddings
    average_layer = Lambda(lambda x: tf.reduce_mean(x, axis=1))

    # Define the second branch: a time-distributed BERT that applies BERT to all articles per day
    bert_branch = MultiTimeDistributed(bert_model)(news_article_input)
    # Use the pooled output (CLS token)
    bert_branch = Lambda(lambda x: x[:, :, :, 0])(bert_branch)


    #Normalize the averaged BERT output
    bert_branch = Normalization(axis=1)(bert_branch)

    # Combine the output of the two branches
    x = concatenate([stock_data_input, bert_branch])

    x = Lambda(lambda t: [t, t])(x)  # Duplicate the input for query and value
    x = Attention()(x)

    # Pass output of transformer/attention layers into LSTM block
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=384, step=32)
    x = LSTM(lstm_units)(x)

    # Pass output of LSTM into dense layers for output
    for i in range(hp.Int('n_layers', 1, 5)):  # Number of Dense layers is a hyperparameter
        layer_units = hp.Int('units_' + str(i), min_value=32, max_value=256, step=32)
        x = Dense(layer_units, activation='relu')(x)
        
    output = Dense(1)(x)

    # Create and compile the model
    model = Model(inputs=[news_article_input, stock_data_input], outputs=output)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['MeanAbsoluteError'])
    return model

#Hyperband Initialization
tuner = Hyperband(
    build_model,
    objective='mean_absolute_error',  # Try to minimize MAE
    max_epochs=15,  # Max number of epochs for each trial
    directory='hyperband_logs',  # Directory for storing logs and models
    project_name='seer_v1_Optimized'  # Name for this set of trials
)


# Generate the combined dataset
stock_names = TotalStockList[:1]#(TotalStockList.index('EIX')+1)] # Grabbing stocks that we have news for, stopping at "EIX"
random.shuffle(stock_names)
print("Loading Datasets Now")
combined_dataset = combine_multiple_stocks(stock_names)


print("Batching Datasets Now")
batch_size = 12 # choose an appropriate batch size
batched_train_dataset = combined_dataset.batch(batch_size)
# batched_validation_dataset = validation_dataset.batch(batch_size)


# Train the model using datasets
print("Beginning Model Training")
tuner.search(batched_train_dataset, epochs=15)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the LSTM layer is {best_hps.get('lstm_units')}, 
the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(batched_train_dataset, epochs=15)

# Model summary
model.summary()

# Save the model
model.save("seer_v1_optimized.h5")
