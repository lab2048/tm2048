# How-to
- Install: `pip install git+https://github.com/lab2048/tm2048.git`

# Description

## TMTool

* Variables
  * `setowords_tw`: StopWords of zh-tw, from https://github.com/lab2048/pyCorpus/raw/main/stopwords_tw.txt"
  * `setowords_cn`: StopWords of zh-cn, from https://github.com/lab2048/pyCorpus/raw/main/stopwords_cn.txt"

* Modules for tokens
  * `deStopWords(words)`:  e.g. `df['tokens'].apply(TMTool.deStopWords)`
  * `dePunctuation(words)`:  e.g., `df['tokens'].apply(TMTool.dePunctuation)`
        
* Modules for text
  * `deEmojify(text)` : e.g., `df['text'].apply(TMTool.deEmojify)`
  * `deSpace(text)`   : e.g., `df['text'].apply(TMTool.deSpace) (Questionable!)`

* Modules for list of tokens
  * `get_word2vec(words)`: e.g., `TMTool.get_word2vec(df['tokens'])`
  * `get_common_words(words)`: e.g., `TMTool.get_common_words(df['tokens'],topN=2000)`
        
* Modules for plotting
  * `plot_w2v_tsne_bokeh(w_f, w2v, title="word2vec")`
    * `w_f`: list of (word, freq) tutple generated from `Counter.most_common()`
    * `w2v`: wordVector generated from `get_word2vec()`

