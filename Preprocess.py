import numpy as np
from sklearn import preprocessing

# Load the data set saved in a csv file
raw_csv_data = np.loadtxt('Business_case_dataset.csv', delimiter=',')
# Remove Id since ID shouldn't affect our model
unscaled_inputs_all = raw_csv_data[:, 1:-1]

# Grab all of our target values
targets_all = raw_csv_data[:, -1]


# Balance out the data. Since we have a lot of people who didn't buy and a little who did purchase after 6 months
# We want to even out how many people purchased and didn't purchase for our model
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

# Use Sklearn in order to preprocess all of our data for us
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# Since time of year can influence when people purhcase books, like during the holidays, we don't want our model to be
# influenced by this. This is why we shuffle our data in the preprocess steps
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

samples_count = shuffled_inputs.shape[0]

# Perform 80 - 10 - 10 split of our data train - validation - test
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count: train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count: train_samples_count + validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
test_targets = shuffled_targets[train_samples_count + validation_samples_count:]

# Save our data into npz files for our model
np.savez('Business_case_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Business_case_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Business_case_data_test', inputs=test_inputs, targets=test_targets)
