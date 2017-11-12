# followed tutorial http://dataaspirant.com/2015/05/25/collaborative-filtering-recommendation-engine-implementation-in-python/

import pandas as pd
import numpy as np
from math import sqrt
import time
import json

# read user data from file
df = pd.read_csv("data/train.csv")
df_2 = pd.read_csv("data/event_attendees.csv")

for i in range(df.shape[0]):
    if df["interested"][i] == 1:
        df.set_value(i, "rating", 8)
    elif df["not_interested"][i] == 1:
        df.set_value(i, "rating", 3)
    elif df["interested"][i] == 0 and df["not_interested"][i] == 0:
        df.set_value(i, "rating", -10)
    else:
        print("error")

pc = df[["user", "event", "rating"]]
pc = pc.applymap(str)

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

    # print("User1: ", user1)
    # print("User2: ", user2)

    user1_sumOfPreferences = sums(both_rated, data, user1)
    user2_sumOfPreferences = sums(both_rated, data, user2)
    # print("Sum: ", user1_sumOfPreferences, user2_sumOfPreferences)

    user1_sumOfSquarePreferences = sumOfSquares(both_rated, data, user1)
    user2_sumOfSquarePreferences = sumOfSquares(both_rated, data, user2)
    # print("Sum of Squares: ", user1_sumOfSquarePreferences, user2_sumOfSquarePreferences)

    cross_sumOfPreferences = crossSums(both_rated, data, user1, user2)
    # print(cross_sumOfPreferences)

    S_user1 = user1_sumOfSquarePreferences - pow(user1_sumOfPreferences, 2) / len(both_rated)
    S_user2 = user2_sumOfSquarePreferences - pow(user2_sumOfPreferences, 2) / len(both_rated)
    S_cross = cross_sumOfPreferences - user1_sumOfPreferences * user2_sumOfPreferences / len(both_rated)
    # print("S values: ", S_user1, S_user2, S_cross)

    if pow(S_user1 * S_user2, 0.5) == 0:
        return 0

    r = S_cross / (pow(S_user1 * S_user2, 0.5))
    # print("r: ", r)
    return r


def sums(both_rated, data, user):
    sum = 0
    for item in both_rated:
        sum += float(data[user][item])
    return sum


def sumOfSquares(both_rated, data, user):
    sum = 0
    for item in both_rated:
        sum += pow(float(data[user][item]), 2)
    return sum


def crossSums(both_rated, data, user1, user2):
    sum = 0
    for item in both_rated:
        sum += float(data[user1][item]) * float(data[user2][item])
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
            if pc > 0:
                for event in data[user2]:
                    if event not in data[user]:
                        sum.setdefault(event, 0)
                        sum2.setdefault(event, 0)
                        sum[event] += float(data[user2][event]) * pc
                        sum2[event] += pc
    rankings = [(total / sum2[item], item) for item, total in sum.items()]
    rankings.sort()
    rankings.reverse()
    recommendation_list = [recommend_item for score, recommend_item in rankings]
    return recommendation_list


def createPlot(x, y, x_scale, y_scale, axis_range, y_label, x_label, title):
    plt.bar(x, y)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.axis(axis_range)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(True)
    plt.title(title)

    plt.savefig('plots/{}.png'.format(title))
    plt.close()


# print("Creating dictionary...")
# t0 = time.time()
# data = create_dict(pc, listOfUsers)
# t1 = time.time()
# print("Time taken: ", t1 - t0)
# print("Finished creating dictionary.")
# with open('output/data.json', 'w') as fp:
#     json.dump(data, fp)

with open('output/data.json', 'r') as fp:
    data = json.load(fp)

# x = list(data.keys())
# y = []
# for user in data:
#     y.append(len(data[user]))
#
# import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#
# for i in range(len(x)):
#     x[i] = int(x[i])
#     y[i] = int(y[i])
#
# print(min(x))
# print(max(x))
# print(min(y))
# print(max(y))
#
# x_scale = "linear"
# y_scale = "linear"
# axis_range = [min(x), max(x), min(y), max(y)]
# y_label = "Number of events"
# x_label = "User"
# title = "Number of events attended by user"
# createPlot(x, y, x_scale, y_scale, axis_range, y_label, x_label, title)

import seaborn as sns


def numOfSimEvents(user1, user2):
    both_rated = {}
    for item in data[user1]:
        if item in data[user2]:
            both_rated[item] = 1
    return len(both_rated)


matrix = np.zeros((len(listOfUsers), len(listOfUsers)))
for i in range(len(listOfUsers)):
    for j in range(len(listOfUsers)):
        matrix[i, j] = numOfSimEvents(listOfUsers[i], listOfUsers[j])

print(matrix)
ax = sns.heatmap(matrix)
dataframe = pd.DataFrame(matrix)
plt.savefig('plots/{}.png'.format("heatmap"))


dataframe.to_csv("output/heatmap.csv")
#
# f, ax = plt.subplots(figsize=(10, 8))
# corr = dataframe.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)



#

# dataset = {
#     'Lisa Rose': {
#         'Lady in the Water': 2.5,
#         'Snakes on a Plane': 3.5,
#         'Just My Luck': 3.0,
#         'Superman Returns': 3.5,
#         'You, Me and Dupree': 2.5,
#         'The Night Listener': 3.0},
#     'Gene Seymour': {'Lady in the Water': 3.0,
#                      'Snakes on a Plane': 3.5,
#                      'Just My Luck': 1.5,
#                      'Superman Returns': 5.0,
#                      'The Night Listener': 3.0,
#                      'You, Me and Dupree': 3.5},
#
#     'Michael Phillips': {'Lady in the Water': 2.5,
#                          'Snakes on a Plane': 3.0,
#                          'Superman Returns': 3.5,
#                          'The Night Listener': 4.0},
#     'Claudia Puig': {'Snakes on a Plane': 3.5,
#                      'Just My Luck': 3.0,
#                      'The Night Listener': 4.5,
#                      'Superman Returns': 4.0,
#                      'You, Me and Dupree': 2.5},
#
#     'Mick LaSalle': {'Lady in the Water': 3.0,
#                      'Snakes on a Plane': 4.0,
#                      'Just My Luck': 2.0,
#                      'Superman Returns': 3.0,
#                      'The Night Listener': 3.0,
#                      'You, Me and Dupree': 2.0},
#
#     'Jack Matthews': {'Lady in the Water': 3.0,
#                       'Snakes on a Plane': 4.0,
#                       'The Night Listener': 3.0,
#                       'Superman Returns': 5.0,
#                       'You, Me and Dupree': 3.5},
#
#     'Toby': {'Snakes on a Plane': 4.5,
#              'You, Me and Dupree': 1.0,
#              'Superman Returns': 4.0}}
# print("calculating k more similar users")
# t0 = time.time()
# print(k_closestUsers(listOfUsers[1], 10, data))
# # print(k_closestUsers("Lisa Rose", 3, dataset))
# t1 = time.time()
# print("Time taken: ", t1 - t0)
# print("finished calculating k more similar users")
# print("Recommending events for user 1...")
# t0 = time.time()
# print(recommendation(listOfUsers[1], data))
# # print(recommendation("Toby", dataset))
# t1 = time.time()
# print("Time taken: ", t1 - t0)
#
#
# print(len(listOfUsers))
