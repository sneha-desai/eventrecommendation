import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import tree
from sklearn.externals import joblib

# feature vector consists of attributes for person and attributes for event
# load the data
users = pd.read_csv('~/event_recommendation/users.csv')
events = pd.read_csv('~/event_recommendation/part_events.csv')

user_col = list(users.columns.values)
event_col = list(events.columns.values)[0:9]

train = pd.read_csv('~/event_recommendation/train.csv')
features = user_col + event_col

n_samples = len(train.index)
inputs = pd.DataFrame(columns=features, index=range(n_samples))

# load history
history = pd.read_csv('~/event_recommendation/event_attendees.csv')

# create an attendance history for each user, based on clusters of events
evt_labels = np.load('cluster_labels.npy')

# get the corresponding event id in
event_ids = events['event_id'].values

events.set_index('event_id')
# users.set_index('user_id')
history.set_index('event')

attendance_col = ['user_id']
for i in range(10):
    attendance_col += [str(i)+'_yes', str(i)+'_no', str(i)+'_maybe', str(i)+'_invited']

user_ids = users['user_id']
print(user_ids)
user_attendance = pd.DataFrame(0, columns=attendance_col, index=range(len(user_ids)))
print(user_ids.values.tolist())
user_attendance.loc[:, 'user_id'] = user_ids.values.tolist()
print(user_attendance.head())


for idx, event_id in enumerate(event_ids):

    cluster = evt_labels[idx]

    yes_list = history[history.event == event_id]["yes"].values[0].split()
    yes_list = list(map(int, yes_list))

    maybe_list = history[history.event == event_id]['maybe'].values[0].split()
    maybe_list = list(map(int, maybe_list))

    no_list = history[history.event == event_id]['no'].values[0].split()
    no_list = list(map(int, no_list))

    invited_list = history[history.event == event_id]['invited'].values[0].split()
    invited_list = list(map(int, invited_list))

    for p in yes_list:
        idx = user_attendance.index[user_attendance.user_id == p].tolist()
        if len(idx) > 0:
            count = user_attendance.iloc[idx][str(cluster) + '_yes'].values # pandas series
            print(count)
            user_attendance.set_value(idx, str(cluster) + '_yes', count[0] + 1)

    for p in no_list:
        idx = user_attendance.index[user_attendance.user_id == p].tolist() # pandas index numeric
        if len(idx) > 0:
            count = user_attendance.iloc[idx][str(cluster) + '_no'].values # pandas series
            print(count)
            user_attendance.set_value(idx, str(cluster) + '_no', count[0] + 1)

    for p in maybe_list:
        idx = user_attendance.index[user_attendance.user_id == p].tolist() # pandas index numeric
        if len(idx) > 0:
            count = user_attendance.iloc[idx][str(cluster) + '_maybe'].values # pandas series
            print(count)
            user_attendance.set_value(idx, str(cluster) + '_maybe', count[0] + 1)

    for p in invited_list:
        idx = np.where(user_attendance.user_id == p)
        idx = user_attendance.index[user_attendance.user_id == p].tolist() # pandas index numeric
        if len(idx) > 0:
            count = user_attendance.iloc[idx][str(cluster) + '_invited'].values # pandas series
            print(count)
            user_attendance.set_value(idx, str(cluster) + '_invited', count[0] + 1)

print(user_attendance.head())

for sample in range(n_samples):

    # get user and event id from training sample
    user_id = train.iloc[sample]['user']
    event_id = train.iloc[sample]['event']

    # get user features for specific user_id change to dict
    user_dict = users[users.user_id == user_id].to_dict()
    user_attendance_dict = user_attendance[user_attendance.user_id == user_id].to_dict()

    # get event features for specific event_id change to dict
    event_dict = events[events.event_id == event_id].to_dict()

    # add both dicts into one dict
    dict = {**user_dict, **event_dict, **user_attendance_dict}

    # add the dict as a dataframe row for the inputs dataframe
    inputs.iloc[sample] = dict

# from pandas to np array, to feed into DecisionTree
x = inputs.as_matrix()
print(x[0:3])

# save as csv, you worked hard!
x.to_csv('input_file.csv')

# create output array
y = train['interested', 'not_interested'].toarray()


T = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split='100')
T.fit(x, y)

test =
T.predict_proba(x)
print('training acc ', T.score(x, y))


