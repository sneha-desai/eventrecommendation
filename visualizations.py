import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *

import networkx as nx

def vis_network():

    social_df = pd.read_csv('~/event_recommendation/user_friends.csv')
    n_users = len(social_df.index)
    G = nx.Graph()
    for i in range(n_users):
        G.add_node(social_df.iloc[i]['user'])

    # add edges
    for i in range(n_users):
        friends_list = str(social_df.iloc[i]['friends']).split()
        for friend in friends_list:
            edge = (social_df.iloc[i]['user'], int(friend))
            G.add_edge(*edge)

    nx.draw(G)

def vis_demographic():

    df = pd.read_csv('~/event_recommendation/users.csv')
    locale = df['locale']
    birthyear = df['birthyear']
    gender = df['gender']
    joined_at = df['joinedAt']
    location = df['location']
    timezone = df['timezone']

    # visualize the distribution
    locale_stats = locale.value_counts().to_dict()
    # TODO: Make it pretty
    sns.barplot(list(locale_stats.keys()), list(locale_stats.values()), palette="BuGn_d")
    plt.show()

    birthyear_stats = birthyear.value_counts().to_dict()
    sns.barplot(list(birthyear_stats.keys()), list(birthyear_stats.values()))
    plt.show()

    gender_stats = gender.value_counts().to_dict()
    sns.barplot(list(gender_stats.keys()), list(gender_stats.values()))
    plt.show()

    location_stats = location.value_counts().to_dict()
    # sns.barplot(list(location_stats.keys()), list(location_stats.values()))
    # plt.show()

def vis_event_attr():

    df = pd.read_csv('~/event_recommendation/events.csv')
    #The first nine columns are event_id, user_id, start_time,
    # city, state, zip, country, lat, and lng.  event_id is the id of the event,
    # and user_id is the id of the user who created the event.  city, state, zip,
    # and country represent more details about the location of the venue (if known)

    # plot the distribution of the 100 most common words among the events


    # for each word, how many events mention it?
    event_counts = [0 for k in range(100)]
    for i in range(1, 101):
        attr = "count_" + str(i)
        count = df[df[attr] > 0][attr]
        event_counts[i] += count

    sns.barplot(range(1, 101), event_counts)
    plt.show()

def vis_clusters

if __name__ == "__main__":
    #vis_network()
    # vis_demographic()
    vis_event_attr()