import tensorflow as tf
import tensorflow_text as text
from datetime import timedelta, date
from header import TotalStockList, filtered_start_date, filtered_end_date, window_size, articles_per_day, max_BERT_sequence_length

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def read_and_tokenize(filepath, tokenizer, max_sequence_length=max_BERT_sequence_length):
    article_text = tf.io.read_file(filepath)
    tokens = tokenizer.tokenize(tf.strings.reduce_join(tf.strings.split(article_text, '\n'), separator=' '))
    tokens = tokens.merge_dims(0, -1)
    tokens = tokens[:max_sequence_length]
    padding = [[0, max_sequence_length - tf.shape(tokens)[0]]]
    tokens = tf.pad(tokens, padding, constant_values=0)
    return tf.cast(tokens, tf.int32)


def generator(max_sequence_length=max_BERT_sequence_length):
    yield from [tf.zeros([max_sequence_length], dtype=tf.int32) for _ in range(articles_per_day)]

def generate_dataset_for_stock(stock_name):
    print(f"Generating dataset for stock: {stock_name}")

    start_date = filtered_start_date.date()
    end_date = filtered_end_date.date()
    tokenizer = text.BertTokenizer("src/BERT_base_uncased/vocab.txt", lower_case=True)
    max_sequence_length = max_BERT_sequence_length  
    date_articles_list = []
    date_list = []
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        filepath_pattern = f'database/{stock_name}/news/{date_str}/article*.txt'
        filepaths = tf.io.gfile.glob(filepath_pattern)
        if not filepaths:
            empty_day_ds = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=[max_sequence_length], dtype=tf.int32)).batch(articles_per_day)
            date_articles_list.append(empty_day_ds)
        else:
            filepaths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
            day_articles_ds = filepaths_ds.map(lambda filepath: read_and_tokenize(filepath, tokenizer, max_sequence_length), num_parallel_calls=tf.data.AUTOTUNE)
            num_articles = tf.data.experimental.cardinality(day_articles_ds).numpy()
            if num_articles < articles_per_day:
                additional_articles = articles_per_day - num_articles
                for _ in range(additional_articles):
                    day_articles_ds = day_articles_ds.concatenate(tf.data.Dataset.from_tensor_slices([tf.zeros([max_sequence_length], dtype=tf.int32)]))
            else:
                day_articles_ds = day_articles_ds.take(articles_per_day)
            day_articles_ds = day_articles_ds.batch(articles_per_day)
            date_articles_list.append(day_articles_ds)
    concat_ds = date_articles_list[0]
    for ds in date_articles_list[1:]:
        concat_ds = concat_ds.concatenate(ds)
    windowed_ds = concat_ds.window(window_size, shift=1, drop_remainder=True)
    windowed_ds = windowed_ds.flat_map(lambda window: window.batch(window_size))
    windowed_ds = windowed_ds.prefetch(tf.data.AUTOTUNE)
    return windowed_ds, date_list

if __name__ == "__main__":
    stock_names = TotalStockList  # Add your stock names here

    for stock_name in stock_names:
        windowed_ds, date_list = generate_dataset_for_stock(stock_name)
        for i, window in enumerate(windowed_ds):
            print(f"Dates for window {i+1}: {date_list[i:i+window_size]}")
            print(window)

