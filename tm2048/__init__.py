import pickle
import pandas as pd
from IPython.display import Markdown, display
pd.options.display.float_format = '{:.4f}'.format

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
