import re

# zh_pat = r'[\u4e00-\u9fff]+'
zh_pat = r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\u2ceb0-\u2ebef\u30000-\u3134f]+'

import time

def timer_decorator(func):
    """
    A decorator that measures the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

# This decorator can be used to measure the execution time of any function.


def read_sheet(sheet_id, sheet_name):
    """
    This function reads a Google Sheet and returns it as a pandas DataFrame.

    Args:
        sheet_id (str): The ID of the Google Sheet. This can be found in the URL of the Google Sheet.
        sheet_name (str): The name of the sheet within the Google Sheet to read.

    Returns:
        DataFrame: The Google Sheet data as a pandas DataFrame.
    """
    # Construct the URL for the Google Sheet
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    # Read the Google Sheet data into a pandas DataFrame
    df = pd.read_csv(url)
    return df




def list_module_functions(module):
    
    """
    The list_module_functions() function takes a module as input 
    and list all functions in the module.
    """
    function_list = [name for name in dir(module) if callable(getattr(module, name))]
    return function_list




if __name__ == "__main__":
    
    import sys
    import pandas as pd
    
    # List all functions in the module
    fs = list_module_functions(sys)
    print(fs)
    # list_module_functions(openai_util)
    
    highlight_matched_word(pattern = r"(?<!不)少年", sentence="現在不少年輕人都喜歡打電動。")
    highlight_matched_word(pattern = r"(?<!不)少年", sentence="人不輕狂枉少年。")
    
    pattern = r"(?<!不)少年"
    sentences = ["現在不少年輕人都喜歡打電動。", "人不輕狂枉少年。"]
    for sentence in sentences:
        if re.search(pattern, sentence):
            highlight_matched_word(pattern = pattern, sentence=sentence)