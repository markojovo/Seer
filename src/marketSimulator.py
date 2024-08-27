import tensorflow as tf
import tensorflow_text as text
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataset_combiner import combine_multiple_stocks
from header import window_size, articles_per_day, TotalTestList, max_BERT_sequence_length, num_cols, MultiTimeDistributed, TotalTradeList, ScalingLayer, custom_loss, custom_binary_loss
from transformers import TFBertModel, BertConfig
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.animation as animation
import random
# matplotlib backend setting for VS Code
import matplotlib
matplotlib.use('Agg')

# Initialize the tokenizer
tokenizer = text.BertTokenizer("src/BERT_base_uncased/vocab.txt", lower_case=True)

profitable_predictions = []
non_profitable_predictions = []




def get_stock_sample_df(stock_name, date):
    # Handle numerical data
    filepath = f'database/{stock_name}/{stock_name}_data.csv'
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    end_idx = df.index.get_loc(date)
    start_idx = max(0, end_idx - window_size)
    numerical_data = np.expand_dims(df.iloc[start_idx:end_idx].drop(columns='Label').values, axis=0) # end_idx is exclusive and Label column is dropped
    # Get the label (the label is the value from the next day)
    growth_col_idx = df.columns.get_loc('Growth')  # Get the column index of 'Growth'
    if end_idx+1 < len(df):
        label = df.iloc[end_idx+1, growth_col_idx]
    else:
        label = np.nan  # if there's no next day data, set the label as NaN

    # Handle news data
    news_data = []
    for i in range(window_size):
        current_date = date - timedelta(days=window_size-1-i)
        date_str = current_date.strftime("%Y-%m-%d")
        filepath_pattern = f'database/{stock_name}/news/{date_str}/article*.txt'
        filepaths = tf.io.gfile.glob(filepath_pattern)

        day_articles = []
        if filepaths:  # Check if filepaths is not empty
            for filepath in filepaths[:articles_per_day]:
                article_text = tf.io.read_file(filepath)
                tokens = tokenizer.tokenize(tf.strings.reduce_join(tf.strings.split(article_text, '\n'), separator=' '))
                tokens = tokens.merge_dims(0, -1)[:max_BERT_sequence_length]
                tokens = tf.pad(tokens, [[0, max_BERT_sequence_length - tf.shape(tokens)[0]]])
                day_articles.append(tokens.numpy())

        # Add zero tensors for missing articles
        while len(day_articles) < articles_per_day:
            tokens = tf.zeros([max_BERT_sequence_length], dtype=tf.int32)
            day_articles.append(tokens.numpy())

        news_data.append(np.array(day_articles))  # append day_articles as numpy array
        
        

    # Convert the list to numpy array and add an extra dimension at the beginning
    news_data = np.expand_dims(np.array(news_data), axis=0)

    return ((news_data, numerical_data), label)

def run_simulation(start_date, end_date, stock_names, threshold, threshold_max = 500, strategy='threshold', num_samples_to_print=5): 
    # Load SPY data
    spy_data = pd.read_csv('database/SPY/SPY_data.csv', parse_dates=True, index_col=0)
    spy_growth = spy_data.loc[start_date:end_date, 'Growth']
    

    model = load_model('seer_v1.h5', custom_objects={'MultiTimeDistributed': MultiTimeDistributed, 'TFBertModel': TFBertModel, 'ScalingLayer':ScalingLayer,'custom_binary_loss':custom_binary_loss, 'custom_loss':custom_loss})

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    spy_values = [1.0]
    actual_labels_cumulative = [1.0]
    current_date = start_date
    win_days = 0  
    loss_days = 0  
    previous_value = 1.0
    model_outputs = []
    actual_labels = []
    absolute_errors = []

    samples_printed = 0

    while current_date <= end_date:
        daily_stocks_to_buy = []
        for stock_name in stock_names:
            try:
                stock_sample = get_stock_sample_df(stock_name, current_date)
            except:
                print(f"Error in retrieving stock data for: {stock_name}, skipping...")
                continue
            (news_data, numerical_data), label = stock_sample
            if numerical_data.shape[1:] != (window_size, num_cols):
                print("Invalid input shape")
                continue

            if label == 0:
                continue

            model_input = [numerical_data]
            model_output = model.predict(model_input)

            if np.isnan(model_output[0][0]) or np.isnan(label):
                print("Result is nan")
                continue

            error = abs(model_output[0][0] - label)

            model_outputs.append(model_output[0][0])
            actual_labels.append(label)
            absolute_errors.append(error)

            if model_output[0][0] > threshold and model_output[0][0] < threshold_max and not np.isnan(model_output[0][0]) and not np.isnan(label):
                daily_stocks_to_buy.append((stock_name, model_output[0][0], label))  

        if daily_stocks_to_buy:
            if strategy == 'top_growth':
                daily_stocks_to_buy.sort(key=lambda x: x[1], reverse=True)  
                stock_name, _, label = daily_stocks_to_buy[0]
                portfolio_value *= (1 + label / 100)
            elif strategy == 'threshold':
                investment_per_stock = portfolio_value/len(daily_stocks_to_buy)
                for stock_name, _, label in daily_stocks_to_buy:
                    portfolio_value += investment_per_stock * (label / 100)
            portfolio_values.append(portfolio_value)

        if portfolio_value > previous_value:
            win_days += 1
        elif portfolio_value < previous_value:
            loss_days += 1

        print(f"Cumulative portfolio value on {current_date.strftime('%Y-%m-%d')}: {portfolio_value*100}%")  # Added line to print the cumulative portfolio value each day


        # Update SPY cumulative returns
        spy_values.append(spy_values[-1] * (1 + spy_growth[current_date.strftime('%Y-%m-%d')] / 100))


        # Update Actual cumulative returns
        actual_labels_cumulative.append(portfolio_value)


        previous_value = portfolio_value

        current_date += timedelta(days=1)

    print(f"Final portfolio value from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}: {portfolio_value*100}%")
    print(f"Total win days: {win_days}")
    print(f"Total loss days: {loss_days}")

    spy_values = spy_values[1:]
    return model_outputs, actual_labels_cumulative, absolute_errors, portfolio_values, spy_values





# Define a function to plot predictions profit vs loss
def plot_profit_vs_loss(profitable_predictions, non_profitable_predictions):
    plt.figure(figsize=(10, 6))
    plt.hist([profitable_predictions, non_profitable_predictions], bins=50, alpha=0.6, label=['Profitable', 'Non-Profitable'], color=['g', 'r'])
    plt.legend(loc='upper right')
    plt.title('Profitable vs Non-Profitable Predictions')
    plt.xlabel('Model Output')
    plt.ylabel('Number of Trades')
    plt.show()


def plot_histogram(data, title):
    plt.figure(figsize=(10, 6))
    mu, std = norm.fit(data)
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = f"{title}: mean = {mu:.2f}, stdv = {std:.2f}"
    plt.title(title)
    plt.show()

def plot_model_output_vs_error(model_outputs, absolute_errors):
    plt.figure(figsize=(10, 6))
    plt.scatter(model_outputs, absolute_errors, alpha=0.5)
    z = np.polyfit(model_outputs, absolute_errors, 2)
    p = np.poly1d(z)
    plt.plot(model_outputs,p(model_outputs),"r--")
    plt.title('Model Output vs Absolute Error')
    plt.xlabel('Model Output')
    plt.ylabel('Absolute Error')
    plt.show()

def plot_model_output_vs_actual_growth(model_outputs, actual_labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(model_outputs, actual_labels, alpha=0.5)
    z = np.polyfit(model_outputs, actual_labels, 3)  # Fit a 3rd degree polynomial
    p = np.poly1d(z)
    plt.plot(model_outputs, p(model_outputs),"r--")
    plt.title('Model Output vs Actual Growth')
    plt.xlabel('Model Output')
    plt.ylabel('Actual Growth')
    plt.show()



def create_plots(model_outputs, actual_labels, absolute_errors):
    plot_histogram(model_outputs, 'Model Outputs')
    plot_histogram(actual_labels, 'Actual Labels')
    plot_histogram(absolute_errors, 'Absolute Errors')
    plot_model_output_vs_error(model_outputs, absolute_errors)
    plot_model_output_vs_actual_growth(model_outputs, actual_labels)
    plot_profit_vs_loss(profitable_predictions, non_profitable_predictions)


def optimize_threshold(start_date, end_date, stock_names, min_threshold=0.6, max_threshold=0.9, step_size=0.1):
    best_threshold = min_threshold
    max_profit = 0.0

    for threshold in np.arange(min_threshold, max_threshold + step_size, step_size):
        _, _, _, final_portfolio_value = run_simulation(start_date, end_date, stock_names, threshold=threshold)
        if final_portfolio_value > max_profit:
            max_profit = final_portfolio_value
            best_threshold = threshold

    print(f"Optimal Threshold for maximum profit: {best_threshold}")
    return best_threshold


def print_last_dataset_samples(dataset, num_samples=5):
    all_samples = list(dataset)  # Convert dataset to list to access elements. This could be memory intensive for large datasets.
    last_samples = all_samples[-num_samples:]  # Select the last num_samples elements.
    for i, ((news_data, numerical_data), label) in enumerate(last_samples, start=1):
        print(f"Sample {i} News Data:\n {news_data}")
        print(f"Sample {i} Numerical Data:\n {numerical_data}")
        print(f"Sample {i} Label:\n {label}\n")

# Create a function to randomly select a start and end date for a month
def random_period_within_range(start_date, end_date, period_length):
    date_range = pd.date_range(start_date, end_date)
    chosen_start = random.choice(date_range.to_list())
    chosen_end = chosen_start + pd.DateOffset(days=period_length)
    
    # Make sure the chosen_end date is not exceeding the end_date.
    if chosen_end > pd.to_datetime(end_date):
        return random_period_within_range(start_date, end_date, period_length)

    return chosen_start.strftime('%Y-%m-%d'), chosen_end.strftime('%Y-%m-%d')






def animate(i):
    ax.clear()
    if i <= len(spy_values):  
        ax.plot(range(i), np.array(actual_labels_cumulative[:i]) * 100, label='Seer')
        ax.plot(range(i), np.array(spy_values[:i]) * 100, label='SPY')
    else:  # if we reached the end of the data, keep plotting the last frame
        
        ax.plot(range(len(actual_labels_cumulative)-1), np.array(actual_labels_cumulative[:-1]) * 100, label='Seer')
        ax.plot(range(len(spy_values)), np.array(spy_values) * 100, label='SPY')

    ax.legend(loc='upper left')
    ax.set_ylim([90, 400])  # 80% to 150% bounds
    ax.set_xlim([0, len(spy_values)-1])  # Total number of days
    ax.set_xlabel('Days')
    ax.set_ylabel('Value (%)')
    ax.set_title(f'Start date: {month_start_date}')




if __name__ == "__main__":
    start_date = '2019-01-01'  
    end_date = '2023-01-30' #'2023-03-02' 
    stock_names = TotalTestList

    #optimal_threshold = optimize_threshold(start_date, end_date, stock_names) #optimal = 0.8
    #model_outputs, actual_labels, absolute_errors, _ = run_simulation(start_date, end_date, stock_names, threshold=0.7, threshold_max = 500, strategy='threshold')
    #create_plots(model_outputs, actual_labels, absolute_errors)
    #print(f"Optimal Threshold: {optimal_threshold}")

    num_simulations = 12
    # Loop the entire process
    for anim_num in range(num_simulations):  # num_simulations is however many simulations you want to run
        month_start_date, month_end_date = random_period_within_range(start_date,end_date, 365)

        # Run simulation for the randomly selected month
        model_outputs, actual_labels_cumulative, absolute_errors, portfolio_values, spy_values = run_simulation(month_start_date, month_end_date, stock_names, threshold=0.75, threshold_max = 500, strategy='threshold')
        print(spy_values)
        print(actual_labels_cumulative)

        fig, ax = plt.subplots()

        # Create an animation
        
        ani = animation.FuncAnimation(fig, animate, frames=len(spy_values) + 30, repeat=False)
        ani.save(f'animations/animationLong{anim_num}.gif', writer='pillow', fps=13)
        
        
