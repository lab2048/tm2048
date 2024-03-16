import re
import pandas as pd
from datetime import datetime
from collections import Counter


import zhon.hanzi
zh_pat = f"[{zhon.hanzi.characters}|{zhon.hanzi.punctuation}]"

def convert_date_string(date_str):
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    if re.match(pattern, date_str):
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return pd.to_datetime(date_str, format='%a %b %d %H:%M:%S %Y', errors='coerce')

def convert_timestamp(post_df):
    post_df['timestamp'] = post_df['ptime'].apply(convert_date_string)
    post_df['month_year'] = post_df['timestamp'].dt.to_period('M')
    post_df['ptime'] = post_df['timestamp'].dt.strftime('%Y-%m-%d')
    return post_df

def filter_post_df(post_df, start_month=None, end_month=None):
    start_date = pd.Period(start_month, freq='M')
    end_date = pd.Period(end_month, freq='M')
    return post_df[(post_df['month_year'] >= start_date) & (post_df['month_year'] <= end_date)]


# Cleaning non-Chinese text and white space

def clean_text(text):
    text = re.sub('[／「」\r\t ]+', '', text)
    text = re.sub("[\n\s]+", " ", text)
    text = re.sub('媒體來源.*?新聞標題', '', text)
    text = re.sub('來源.*?新聞標題', '', text)
    text = re.sub('備註請放最後面違者新聞文章刪除|完整新聞連結或短網址需放媒體原始連結，不可用轉載媒體連結備註', '', text)
    text = re.sub("完整.*?報導", '', text)
    text = re.sub('記者.*?報導', '', text)
    text = re.sub('路透|快訊|完整新聞內文|年月日', '', text)
    text = re.sub("（）|〔〕|’’|“”", '', text)
    text = re.sub("【.*?】|《.*?》|〔.*?〕", '', text)
    text = "".join(re.findall(zh_pat, text))
    return text


def clean_post_process(post_df):
    post_df = convert_timestamp(post_df)
    post_df = filter_post_df(post_df)
    post_df["cleaned"] = post_df["pcontent"].apply(clean_text)
    print(post_df.month_year.value_counts().sort_index())
    return post_df



def observe_post(post_df):
    print(post_df.shape)
    vis.plot_counter(Counter([len(c) for c in post_df['cleaned']]), gap=100)
    print(max([len(c) for c in post_df['cleaned']]))
    print(Counter([len(c) for c in post_df['cleaned']]).most_common(20))


# Merging post and comments data

def merge_data(post_df, comments_df):
    merged_df = post_df.merge(comments_df[['plink', 'commentor', 'ccontent']], on='plink', how='left')
    post_comment_counts = merged_df["plink"].value_counts().reset_index()
    post_comment_counts.columns = ["plink", "comment_count"]
    merged_df = pd.merge(merged_df, post_comment_counts, on="plink", how="left")
    return merged_df[['plink', 'cleaned', 'poster', 'ptitle', 'ptime', 'commentor', 'ccontent']]
