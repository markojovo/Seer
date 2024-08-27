import tensorflow as tf
from numerical_dataset_generator import generate_stock_dataset
from news_dataset_generator import generate_dataset_for_stock
from header import TotalStockList

def combine_datasets(stock_name):
    numerical_dataset = generate_stock_dataset(stock_name)
    return numerical_dataset
    news_dataset, _ = generate_dataset_for_stock(stock_name)

    if numerical_dataset is not None:
        # Change the format to ((news_data, numerical_data), label)
        combined_dataset = tf.data.Dataset.zip((news_dataset, numerical_dataset))
        combined_dataset = combined_dataset.map(lambda x, y: ((x, y[0]), y[1]))
        return combined_dataset
    else:
        return None

def combine_multiple_stocks(stock_names):
    # Create list to store the datasets
    datasets = []
    for stock_name in stock_names:
        try:
            dataset = combine_datasets(stock_name)
            if dataset is not None:
                datasets.append(dataset)
        except:
            continue
        

    # Concatenate all datasets into one
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(dataset)


    # Add the filter to remove instances with label 0.0
    print("BEGINNING FILTERING OF DATASET")
    combined_dataset = combined_dataset.filter(lambda x, y: tf.not_equal(y, 0.0))
    print("FILTERING OF DATASET COMPLETE") 

    combined_dataset = combined_dataset.shuffle(buffer_size=5000)

    return combined_dataset

if __name__ == "__main__":
    stock_names = ['A','AAL']  # Add your stock names here
    combined_dataset = combine_multiple_stocks(stock_names)

    # Print out first 5 examples from the combined dataset
    print("First 5 examples from the combined dataset:")
    for example in combined_dataset.take(10):
        print(example)
