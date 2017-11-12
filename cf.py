# followed tutorial http://dataaspirant.com/2015/05/25/collaborative-filtering-recommendation-engine-implementation-in-python/

import pandas as pd
import numpy as np
from math import sqrt

# read user data from file
df = pd.read_csv("data/train.csv")

for i in range(df.shape[0]):
    if df["interested"][i] == 1:
        df.set_value(i, "rating", 1)
    elif df["not_interested"][i] == 1:
        df.set_value(i, "rating", 0)
    elif df["interested"][i] == 0 and df["not_interested"][i] == 0:
        df.set_value(i, "rating", -1)
    else:
        print("error")

pc = df[["user", "event", "rating"]]

listOfUsers = list(pc["user"].unique())


def create_dict(pc, listOfUsers):
    pc_dict = {}
    for i in range(len(listOfUsers)):
        events = {}
        for j in range(pc.shape[0]):
            if pc["user"][j] == listOfUsers[i]:
                events[pc["event"][j]] = pc["rating"][j]
        pc_dict[listOfUsers[i]] = events
    return pc_dict


def pearson_correlation(user1, user2, data):
    both_rated = {}
    for item in data[user1]:
        if item in data[user2]:
            both_rated[item] = 1
    if len(both_rated) == 0:
        return 0
    user1_sumOfPreferences = sums(both_rated, data, user1)
    user2_sumOfPreferences = sums(both_rated, data, user2)

    user1_sumOfSquarePreferences = sumOfSquares(both_rated, data, user1)
    user2_sumOfSquarePreferences = sumOfSquares(both_rated, data, user2)

    cross_sumOfPreferences = crossSums(both_rated, data, user1)

    S_user1 = user1_sumOfSquarePreferences - pow(user1_sumOfPreferences, 2) / len(both_rated)
    S_user2 = user2_sumOfSquarePreferences - pow(user2_sumOfPreferences, 2) / len(both_rated)
    S_cross = cross_sumOfPreferences - user1_sumOfPreferences * user2_sumOfPreferences / len(both_rated)

    if pow(S_user1 * S_user2, 0.5) == 0:
        return 0

    r = S_cross / (pow(S_user1 * S_user2, 0.5))
    return r


def sums(both_rated, data, user):
    sum = 0
    for item in both_rated:
        sum += data[user][item]
    return sum


def sumOfSquares(both_rated, data, user):
    sum = 0
    for item in both_rated:
        sum += pow(data[user][item], 2)
    return sum


def crossSums(both_rated, data, user):
    sum = 0
    for item in both_rated:
        sum += pow(data[user][item], 2)
    return sum


def k_closestUsers(user, k, data):
    scores = [(pearson_correlation(user, other_user, data), other_user) for other_user in data if
              other_user != user]
    scores.sort()
    scores.reverse()
    return scores[0:k]


def recommendation(user, data):
    sum = {}
    sum2 = {}
    rankings = []
    for user2 in data:
        if user2 != user:
            pc = pearson_correlation(user, user2, data)
            for event in data[user2]:
                if event not in data[user]:
                    sum.setdefault(event, 0)
                    sum2.setdefault(event, 0)
                    sum[event] += data[user2][event] * pc
                    sum2[event] += pc
    rankings = [(total / sum2[item], item) for item, total in sum.items()]
    rankings.sort()
    rankings.reverse()
    recommendation_list = [recommend_item for score, recommend_item in rankings]
    return recommendation_list


print("Creating dictionary...")
data = create_dict(pc, listOfUsers)
print(data)
print("Finished creating dictionary.")
data_df = pd.DataFrame.from_dict(data)
data_df.to_csv("output/data.csv")
print("calculating k more similar users")
print(k_closestUsers(listOfUsers[0], 5, data))
print("finished calculating k more similar users")
print("Recommending events for user 1...")
print(recommendation(listOfUsers[0], data))

