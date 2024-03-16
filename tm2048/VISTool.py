import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def highlight_matched_word(pattern, sentence):
    """
    The highlight_matched_word() function takes a pattern and a sentence as input 
    and highlights the matched word.
    """
    highlighted_sentence = re.sub(pattern, r'\033[1m\033[91m\g<0>\033[0m', sentence) # red word
    # highlighted_sentence = re.sub(pattern, r'\033[1m\033[43m\g<0>\033[0m', sentence) # yellow backgorund
    print(highlighted_sentence)

def plot_counter_plt(counter):
    # Extract the keys and values from the Counter object
    keys = list(counter.keys())
    values = list(counter.values())

    # Set the figure size with a 1:2 height to width ratio
    plt.figure(figsize=(8, 4))

    # Plot the bar chart
    plt.bar(keys, values)

    # Set labels and title
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Counter Result")

    # Show the plot
    plt.show()

def plot_counter(counter, gap=5):
    # Convert the Counter to a pandas DataFrame
    counter_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    counter_df.columns = ['Value', 'Count']

    # Set the figure size with a 1:2 height to width ratio
    plt.figure(figsize=(8, 4))

    # Create the bar plot using seaborn
    ax = sns.barplot(x='Value', y='Count', data=counter_df)
    ax.set_xticks(ax.get_xticks()[::gap])

    # Set labels and title
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Counter Result")

    # Show the plot
    plt.show()

def plot_year_distribution(data, variable):
    # Calculate the distribution of each year
    year_counts = data[variable].value_counts().sort_index()

    plt.figure(figsize=(8, 4))

    # Plot the bar chart for the distribution of each year using seaborn
    ax = sns.barplot(x=year_counts.index, y=year_counts.values)
    ax.set_xticks(ax.get_xticks()[::5])
    # ax.set_xticklabels(ax.get_xticks()[::5])
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title(f"Distribution of {variable}")
    plt.show()