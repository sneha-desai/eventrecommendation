import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import decomposition

def get_tfidf():

    df = pd.read_csv('~/event_recommendation/events.csv')
    print(df.columns)
    # for each word, how many events mention it?
    event_counts = [0 for k in range(100)]
    for i in range(100):
        attr = "c_" + str(i+1)
        count = len(df[df[attr] > 0].index)
        event_counts[i] = count

    n_events = len(df.index)
    # compute tf-idf
    idf_list = []
    for word in range(100):
        attr = "count_" + str(word+1)
        idf_list.append(n_events / (np.log(event_counts[word] + 1)))


    for event in range(n_events):
        for idx, idf in enumerate(idf_list):
            attr = "c_" + str(idx+1)
            df.loc[event, attr] *= idf

    df.to_csv('event_tfidf.csv')

def pca():


if __name__=="__main__":
    pca()
