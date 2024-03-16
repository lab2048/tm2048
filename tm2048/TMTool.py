#! /usr/bin/env python

import re
import pkg_resources
from urllib.request import urlopen
import unicodedata # for removing Chinese puctuation
from collections import Counter
from gensim.models import Word2Vec


from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d, LinearColorMapper, LogColorMapper
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
output_notebook()

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import time

class TMTool:
    
    def __init__(self):
        # self.stopwords_tw = self.__load_stopwords_url("https://github.com/lab2048/pyCorpus/raw/main/stopwords_tw.txt")
        # self.stopwords_cn = self.__load_stopwords_url("https://github.com/lab2048/pyCorpus/raw/main/stopwords_cn.txt")
        self.stopwords_tw = self.__load_stopwords_file("stopwords_tw.txt")
        self.stopwords_cn = self.__load_stopwords_file("stopwords_cn.txt")

    """
    The highlight_matched_word() function takes a pattern and a sentence as input 
    and highlights the matched word.
    """
    @staticmethod
    def highlight_matched_word(pattern, sentence):
        highlighted_sentence = re.sub(pattern, r'\033[1m\033[91m\g<0>\033[0m', sentence) # red word
        # highlighted_sentence = re.sub(pattern, r'\033[1m\033[43m\g<0>\033[0m', sentence) # yellow backgorund
        print(highlighted_sentence)

    @staticmethod
    def label_text(text):
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        mentions = re.findall(r'@\w+', text)
        hashtags = re.findall(r'#\w+', text)
        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        html_tags = re.findall(r'<[^>]+>', text)
        return {
            'emails': emails,
            'mentions': mentions,
            'hashtags': hashtags,
            'links': links,
            'html_tags': html_tags,
        }

    # def __load_stopwords_file(self, filename):
    #     with open(f'corpus/{filename}', 'r', encoding='utf-8') as file:
    #         stopwords = file.read().split("\n")[1:]
    #     return stopwords
    
    def __load_stopwords_file(self, filename):
        resource_path = f'data/{filename}'
        stopwords_content = pkg_resources.resource_string(__name__, resource_path)
        stopwords = stopwords_content.decode('utf-8').splitlines()
        return stopwords[1:]            
    
    def __load_stopwords_url(self, url):            
        text = urlopen(url).read().decode('utf-8')
        return text.split("\n")[1:]     
        
    def deStopWords(self, words):
        out = []
        for word in words:
            if (word not in self.stopwords_tw) and (word not in self.stopwords_cn):
                out.append(word)
        return out
    
    def jieba_tokenize(self, text):
        import jieba
        jieba.load_userdict("dict.txt")
        return list(jieba.cut(text, HMM=True))
    
    def jieba_tagger(self, text):
        import jiebaw
        # incomplete

    @staticmethod
    def dePunctuation(words):
        out = []
        for word in words:
            if word != " " and not unicodedata.category(word[0]).startswith('P'):
                out.append(word)
        return out
    
    
    @staticmethod
    def deEmojify(text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)
    
    @staticmethod
    def deSpace(text_list):
        return text_list.replace("\s", " ", regex=True)

    @staticmethod
    def get_common_words(tokens_list, min_token_len=1, percent=None, topN=None):
        word_count = Counter()
        for tokens in tokens_list:
            for tok in tokens:
                if len(tok) >= min_token_len:
                    word_count[tok] += 1
        if topN is not None:
            return word_count.most_common(topN)
        if percent is not None:
            split = int(len(word_count)*percent)
            return word_count.most_common(split)
        
    def extract_chinese(text):
        """
        This function cleans the input text by removing non-Chinese characters and unnecessary white spaces.
        It was originally copied from the notebook youth_news_0530.ipynb.

        Args:
            text (str): The input string that needs to be cleaned.

        Returns:
            str: The cleaned string with only Chinese characters and necessary white spaces.
        """
        # Replace multiple occurrences of carriage return, tab, and space characters with a single space
        text = re.sub("[\r\t ]+", " ", text)
        
        # Use a regular expression to find all Chinese characters in the text
        # and join them together into a single string
        text = "".join(re.findall(zh_pat, text))
        return text

    
    
    @staticmethod
    def get_word2vec(tokens_list, size=100, window=5, min_count=5):
        model = Word2Vec(tokens_list, min_count=min_count, size=size, window=window, sg=0, workers=4)
        return model

# """ Visual tools
#     plot_w2v_tsne_bokeh()
#         w_f: tuple(word, freq) to filter word to display
#         w2v: word embedding by gensim.models.Word2Vec
# """ 
    @staticmethod
    def plot_w2v_tsne_bokeh(w_f, w2v, title_text="word2vec"): 
        word_freq = [(w, f) for w, f in w_f if len(w) > 1]

        wv  = [w2v[w] for w, f in word_freq]

        stime = time.time()
        print("... dimension deduction by tsne ...")
        tsne = TSNE(n_components=2, random_state=0)
        points = tsne.fit_transform(wv)
        print("... tsne complete in %f ..."%(time.time() - stime))

        plot_df = pd.DataFrame(points, columns = ['x', 'y'])
        plot_df['label'] = [w for w, f in word_freq]
        plot_df['freq'] = [f for w, f in word_freq]

        color_mapper = LogColorMapper(palette='Plasma256', low=min(plot_df['freq']), high=max(plot_df['freq']))

        p = figure(title = title_text)
        p.circle(plot_df["x"], plot_df["y"], fill_alpha=0.2, size=10)
        p.sizing_mode = 'scale_width'
        labels = LabelSet(x='x', y='y', text='label', 
                          x_offset=5, y_offset=5, 
                          text_color={'field': 'freq', 'transform': color_mapper},
                          text_alpha=0.6,
                          source=ColumnDataSource(plot_df), render_mode='canvas')
        p.add_layout(labels)
        show(p)


# """ OTHER TOOLS
#     print_now(): print curretn time
    
# """
    @staticmethod   
    def print_now():
        from datetime import datetime
        print("FINISH at", datetime.now().strftime("%H:%M:%S"))
        
    # df['color'] = df['cluster'].apply(lambda x:generate_color(x, true_k))        
    @staticmethod
    def generate_color(x, total):
        def color_red_blue(ctr, reverse=False):
            r = int(ctr*255)
            b = 255-int(ctr*255)
            if reverse:
                return "#%02x%02x%02x" % (b, 0, r)
            else:
                return "#%02x%02x%02x" % (r, 0, b)
        return color_red_blue(x/total)




if __name__ == "__main__":
    # Example text with HTML tags
    example_text = """
    For more information, contact <a href="support@example.com">support@example.com</a>.
    Follow us on Twitter: @example_user. Visit our website: https://www.example.com
    Join the conversation with #PythonProgramming.
    Connect with John Doe at john.doe@example.com.
    I love using Python for <b>natural language processing</b> and sentiment analysis!
    """

    # Filter information from the text
    filtered_info = filter_text(example_text)

    # Display the filtered information
    print("Emails:", filtered_info['emails'])
    print("Mentions:", filtered_info['mentions'])
    print("Hashtags:", filtered_info['hashtags'])
    print("Links:", filtered_info['links'])
    print("HTML Tags:", filtered_info['html_tags'])
    
    # __all__ = ["deEmojify", "dePunctuation", "deStopWords", "stopwords_cn", "stopwords_tw", "get_common_words"]

# with open("../../pybin/stopwords_tw.txt", encoding="utf-8") as fin:
#     stopwords_tw = fin.read().split("\n")[1:]

# with open("../../pybin/stopwords_cn.txt", encoding="utf-8") as fin:
#     stopwords_cn = fin.read().split("\n")[1:]
