#First tensorflow program

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

california_housing_dataframe = pd.read_csv("http://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"]/= 1000.0

print(california_housing_dataframe)

print(california_housing_dataframe.describe()) #Print some useful statistics about the features

# We want to predict median_house value, that will be our target

# Our input feature will be total rooms

my_feature = california_housing_dataframe[["total_rooms"]]



feature_columns = [tf.feature_column.numeric_column("total_rooms")]
targets = california_housing_dataframe ["median_house_value"]

#We use a linear regressor model

# We choose a linear regressor searching for the minimum through gradient GradientDescentOptimizer
# The learning rate defines the size of each step of the gradient

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

# Gradient clipping ensures the magnitude of the gradients do not become too large during training, which can cause gradient descent to fail.
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)


linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer = my_optimizer)

#Define the input function, which tells TF how to preprocess, batch, shuffle and repeat data for training the classifier
'''
    Parameters:
        features: Dataframe of features
        targets: Dataframe of targets (median_house_value)
        batch_size: Size of batches to be passed to the model
        shuffle: True or False -> Whether to shuffle the data or not
        num_epochs: Number of epochs for which data should be repeated

    Return:
        Tuple of (features, labels) for next data batch
'''

def my_input_fn (features, targets, batch_size=1, shuffle=True, num_epochs=None):
    #Convert pandas data into a dict of np arrays
    features = {key:np.array(value) for key, value in dict(features).items()}

    #Construct a dataset and configure batching/repeating
    ds=Dataset.from_tensor_slices((features,targets))
    ds=ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


#Training the models in 100 steps

_ = linear_regressor.train(input_fn=lambda:my_input_fn(my_feature, targets), steps=100)

prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

#Making the predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print ("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print ("Root Mean Squared Error (on training data): %0.3f" %root_mean_squared_error)
