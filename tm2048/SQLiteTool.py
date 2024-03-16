"""
@author: Jilung
@date: 2023-09-09
@purpose: to query sqlite database
"""

import re
import sqlite3
import pandas as pd
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


'''
@class: SQLiteTool
@purpose: to query sqlite database
'''

class SQLiteTool:
    def __init__(self, database_path: str):
        self.conn = sqlite3.connect(database_path)
        self.conn.create_function("REGEXP", 2, self.regexp)
        self.cur = self.conn.cursor()

    def list_tables(self) -> List[str]:
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in self.cur.fetchall()]
        return tables

    def list_columns(self, table_name: str) -> List[str]:
        self.cur.execute(f"PRAGMA table_info({table_name})")
        columns_info = self.cur.fetchall()
        column_names = [column[1] for column in columns_info]
        return column_names
    
    # def list_columns(self, table_name: str) -> List[Tuple[str, str]]:
    def list_columns_with_type(self, table_name: str) -> List[Tuple[str, str]]:
        self.cur.execute(f"PRAGMA table_info({table_name})")
        columns_info = self.cur.fetchall()
        column_names = [(column[1], column[2]) for column in columns_info]
        return column_names
    
    def get_table_size(self, table_name: str) -> int:
        # First, check if table exists
        self.cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if self.cur.fetchone()[0] == 1:
            # Query to get the size of the table (i.e., the number of rows)
            self.cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            table_size = self.cur.fetchone()[0]
            return table_size
        else:
            return f"Table '{table_name}' does not exist."    
    
    # list the first 5 rows of all columns and return dataframe
    
    def get_columns(self, table_name: str, column_names) -> pd.DataFrame:
        columns_str = ', '.join(column_names)
        query = f"SELECT {columns_str} FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_all(self, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def regexp(self, expr, item):
        try:
            reg = re.compile(expr)
            return reg.search(item) is not None
        except Exception as e:
            # print(f"Error in regexp function: {e}")
            return False

    def query_with_regexp(self, table_name: str, column_name: str, pattern: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} WHERE {column_name} REGEXP ?"
        df = pd.read_sql_query(query, self.conn, params=(pattern,))
        return df
    
    def extract_matching_records(table_name: str, col_name: str, pattern: str) -> List[Tuple]:
        self.cur.execute(f"SELECT * FROM {table_name} WHERE {col_name} REGEXP ?", (pattern,))
        matching_records = self.cur.fetchall()
        
        return matching_records
    
    def add_date(self):
        df = pd.read_sql_query("SELECT * from posts", self.conn)
        df['timestamp'] = pd.to_datetime(df['ptime'], format='%a %b %d %H:%M:%S %Y', errors='coerce')
        print("timestamp.isna(): ", df.timestamp.isna().sum()) # 41 very few
        df['date_str'] = df['timestamp'].dt.strftime('%Y%m%d').fillna('0')
        df['date'] = df['date_str'].astype(int)
        df['timestamp'] = df['timestamp'].astype(str)
        # df['date'].replace(0, np.nan, inplace=True)
        print("date.isna():", df.date.isna().sum()) # 41 very few
        df.drop('date_str', axis=1, inplace=True)
        df.to_sql('posts', self.conn, if_exists='replace', index=False)



    def count_null_values(self, table_name: str, column_name: str) -> int:
        query = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
        self.cur.execute(query)
        result = self.cur.fetchone()
        return result[0]
    
    def count_records(self, table_name: str, column_name: str, value):
        try:
            self.cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} = ?", (value,))
            count = self.cur.fetchone()[0]        
            print(f"The number of records in {table_name} where {column_name} = {value} is: {count}")            
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")


    def dbhead(self, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} Limit 5"
        df = pd.read_sql_query(query, self.conn)
        return df


    def delete_records(self, table_name: str, column_name: str, value):
        try:
            self.cur.execute(f"DELETE FROM {table_name} WHERE {column_name} = ?", (value,))
            self.conn.commit()
            print(f"Successfully deleted records from {table_name} where {column_name} = {value}.")
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")


    def filter_by_date(self, table_name: str, start_date: int, end_date: int) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} WHERE date BETWEEN {start_date} AND {end_date}"
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def filter_comments_by_post_date(self, start_date, end_date):
        query = """
        SELECT c.*
        FROM comment AS c
        JOIN post AS p ON c.post_id = p.post_id
        WHERE p.date >= ? AND p.date <= ?;
        """
        self.cur.execute(query, (start_date, end_date))
        return self.cur.fetchall()
    
    def filter_records(self, table_name: str, column_name: str, filter_list: List[Union[str, int]]) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} WHERE {column_name} IN ({','.join(['?' for _ in filter_list])})"
        result_df = pd.read_sql_query(query, self.conn, params=filter_list)
        return result_df

    def filter_records_batch(self, table_name: str, column_name: str, filter_list: List[Union[str, int]], batch_size=1000) -> pd.DataFrame:
        result_df = pd.DataFrame()
        for i in range(0, len(filter_list), batch_size):
            print(i)
            batch_values = filter_list[i:i + batch_size]
            query = f"SELECT * FROM {table_name} WHERE {column_name} IN ({','.join(['?' for _ in batch_values])})"
            batch_df = pd.read_sql_query(query, self.conn, params=batch_values)
            result_df = pd.concat([result_df, batch_df], ignore_index=True)
        return result_df

    def check_duplicate(self, table_name: str):
        # Get the column names dynamically
        self.cur.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in self.cur.fetchall()]
        
        # Create the SQL query for detecting duplicate rows
        column_str = ", ".join(columns)
        query = f"SELECT {column_str}, COUNT(*) FROM {table_name} GROUP BY {column_str} HAVING COUNT(*) > 1"

        self.cur.execute(query)
        rows = self.cur.fetchall()
        return rows

    
    def remove_duplicate(self, table_name: str):
        self.cur.execute(f"CREATE TABLE temp_table AS SELECT DISTINCT * FROM {table_name}")
        self.cur.execute(f"DROP TABLE {table_name}")
        self.cur.execute(f"ALTER TABLE temp_table RENAME TO {table_name}")
        self.conn.commit()

    def check_duplicate_by_column(self, table_name: str, column_name: str):
        query = f"SELECT {column_name}, COUNT(*) FROM {table_name} GROUP BY {column_name} HAVING COUNT(*) > 1"
        # query = f"SELECT {column_name}, COUNT(*) FROM {table_name} GROUP BY {column_name} HAVING COUNT(*) > 1"
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return rows
    
    def get_duplicate_rows_by_column(self, table_name: str, column_name: str):
        query = f"SELECT * FROM {table_name} WHERE {column_name} IN (SELECT {column_name} FROM {table_name} GROUP BY {column_name} HAVING COUNT(*) > 1)"
        self.cur.execute(query)
        rows = self.cur.fetchall()
        df = pd.DataFrame(rows, columns=[desc[0] for desc in self.cur.description])
        return df
    
    def list_functions(self):
        methods = [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith("__")]
        return methods
    
    def remove_duplicate_by_columns(self, table_name: str, columns: list):
        # Convert list of columns to a comma-separated string
        columns_str = ', '.join(columns)
        
        # Create a temporary table with distinct rows based on specific columns
        self.cur.execute(f"CREATE TABLE temp_table AS SELECT * FROM {table_name} GROUP BY {columns_str}")
        self.cur.execute(f"DROP TABLE {table_name}")
        self.cur.execute(f"ALTER TABLE temp_table RENAME TO {table_name}")
        
        self.conn.commit()
        
        
    
    def plot_monthly(self):
        df = pd.read_sql_query("SELECT timestamp from posts", self.conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        monthly_counts = df.resample('M', on='timestamp').size().reset_index(name='Counts')
        plt.figure(figsize=(10, 6))
        plt.bar(monthly_counts['timestamp'], monthly_counts['Counts'], width=20)
        plt.xticks(rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.show()
        
    def plot_monthly_plotly(self):
        df = pd.read_sql_query("SELECT timestamp from posts", self.conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        monthly_counts = df.resample('M', on='timestamp').size().reset_index(name='Counts')

        # 使用Plotly創建互動式圖表
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=monthly_counts['timestamp'],
            y=monthly_counts['Counts'],
            marker=dict(color='rgb(55, 83, 109)'),  # 可以調整顏色
        ))

        # 更新圖表佈局
        fig.update_layout(
            title='每月資料筆數',
            xaxis=dict(
                title='月份',
                # tickangle=-45,  # X軸標籤傾斜以避免重疊
                type='date',  # 確保x軸以日期格式展示
                dtick="M1",  # X軸刻度間隔設為1個月
                tickformat="%Y-%m",  # X軸日期格式
            ),
            yaxis=dict(title='筆數'),
            bargap=0.1,  # 條形間距
        )
        fig.update_xaxes(
            tickvals=monthly_counts['timestamp'],  # 確定刻度位置
            ticktext=[date.strftime('%Y-%m') for date in monthly_counts['timestamp']]  # 自定義刻度標籤格式
        )

        fig.show()

    def top_n_values(self, table_name: str, column_name: str, n: int, reverse: bool = False):
        order = "DESC" if reverse else "ASC"
        query = f"SELECT {column_name} FROM {table_name} ORDER BY {column_name} {order} LIMIT {n}"
        self.cur.execute(query)
        results = self.cur.fetchall()
        return [result[0] for result in results]

    def update_comment_count(self):
        
        self.cur.execute("PRAGMA table_info(videos)")
        if 'nComment' not in [column[1] for column in self.cur.fetchall()]:
            self.cur.execute("ALTER TABLE videos ADD COLUMN nComment INTEGER DEFAULT 0")
            self.conn.commit()
        
        self.cur.execute("""
        SELECT videoId, COUNT(*) as nComment
        FROM comments
        GROUP BY videoId
        """)

        video_comment_counts = self.cur.fetchall()

        # 更新videos table中的nComment列
        for video_id, count in video_comment_counts:
            self.cur.execute("""
            UPDATE videos
            SET nComment = ?
            WHERE video_id = ?
            """, (count, video_id))

        # self.cur.execute("""
        # UPDATE videos
        # SET comment_count = 0
        # WHERE video_id NOT IN (SELECT DISTINCT videoId FROM comments)
        # """)

        self.conn.commit()    
    
    def update_chat_count(self):
        
        self.cur.execute("PRAGMA table_info(videos)")
        if 'nChat' not in [column[1] for column in self.cur.fetchall()]:
            self.cur.execute("ALTER TABLE videos ADD COLUMN nChat INTEGER DEFAULT 0")
            self.conn.commit()
        
        self.cur.execute("""
        SELECT videoId, COUNT(*) as nChat
        FROM chats
        GROUP BY videoId
        """)

        video_chat_counts = self.cur.fetchall()

        # 更新videos table中的nComment列
        for video_id, count in video_chat_counts:
            self.cur.execute("""
            UPDATE videos
            SET nChat = ?
            WHERE video_id = ?
            """, (count, video_id))

        # self.cur.execute("""
        # UPDATE videos
        # SET comment_count = 0
        # WHERE video_id NOT IN (SELECT DISTINCT videoId FROM comments)
        # """)

        self.conn.commit()    
    
    
    
    def close(self):
        self.conn.close()

