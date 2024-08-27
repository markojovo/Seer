import tensorflow as tf
import pandas as pd
from header import window_size, TotalStockList, filtered_start_date, filtered_end_date, num_cols

def parse_fn(line, stock_name):
    types = [tf.constant([''], dtype=tf.string)] + [tf.constant([0.0], dtype=tf.float32)]*(num_cols+1)
    fields = tf.io.decode_csv(line, record_defaults=types)
    data = tf.stack(fields[1:])  # Collect all data, including "Label"
    return data, stock_name


def windowed_dataset(series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]))  # Next day's "Label" as label
    return dataset

def generate_stock_dataset(stock_name):
    filepath = f'database/{stock_name}/{stock_name}_data.csv'
    print("Running for: "+filepath)

    try:
        dataset = tf.data.TextLineDataset(filepath).skip(1)
        dataset = dataset.map(lambda line: parse_fn(line, stock_name))
        all_features = []
        for features, _ in dataset:
            all_features.append(features)
        windowed_ds = windowed_dataset(all_features, window_size)
        return windowed_ds
    except (tf.errors.InvalidArgumentError, ValueError) as e:
        print(f"Skipping {stock_name} due to error: {e}")
        return None

if __name__ == "__main__":
    stock_names = ['A']  #TotalStockList  # Add your stock names here

    # Use Dataset API to create a dataset of stock_name: file_dataset pairs
    datasets = {stock_name: generate_stock_dataset(stock_name) for stock_name in stock_names}

    # Print out first 5 examples from each dataset
    for stock_name, dataset in datasets.items():
        print(f"First 5 examples for {stock_name}:")
        if dataset is not None:
            for example in dataset.take(5):
                print(example)
