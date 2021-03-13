import pandas as pd
path = r'C:\Users\omgit\PycharmProjects\185cMidterm\Business_case_dataset.csv'
features = ['id', 'BLoverall', 'BLavg', 'price_overall', 'price_avg', 'review', 'review score', 'minutes listened',
            'completion', 'Support Request', 'Last visited minus purchase date', 'targets']
dataset = pd.read_csv(path, names=features)
print(dataset)

dropped_dataset = dataset.drop(['review score'], axis=1)

dropped_dataset.to_csv('Modify_Business_case_dataset', index=False, header=False)
