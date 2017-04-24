import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
import CustomTransformers as Ct
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import CustomSelector as Cs


def load_housing_data(path='/home/datadrive/PythonDev/DeepLearningPython/MLWithScikitLearnAndTensorFlow/Data/housing.csv'):
    return pd.read_csv(path)

housing = load_housing_data()
housing.head()

# To ensure fair distribution between training and test sets
# to avoiding the risk of introducing significant sampling bias
# Use stratified sampling (where data is divided into

# Say the median income is a very important attribute to predict median housing prices.
# We may need to ensure the test set is representative of the the various
# categories of income in the whole dataset.
# Since Median income in the dataset is a continuous numerical attribute. We will need to
# first crete an Income category attribute.

# looking at the Median income histogram, most median income values
# are clustered around 2-5(tens of thousands of dollars). Some going beyond 6a.

# Don't have too many strata and each stratum should be large enough.

# Lets create income category attribute.
# divide median income by 1.5 to limit number of categories.
# Then merge all categories > 5 into category 5

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

# Stratify based on the new income category
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# remove the income_cat attribute so the data is back i it's original
# state. we have the train and test set.
for aset in (strat_train_set, strat_test_set):
    aset.drop(['income_cat'], axis=1, inplace=True)

housing_copy = strat_train_set.copy()
housing_copy = strat_train_set.drop('median_house_value', axis=1)
housing_copy_labels = strat_train_set['median_house_value'].copy()


# housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, s=housing['population'] / 100,
#                   label='Population',
#                   c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()
# plt.show()
# print('TEst')
# corr_matrix = housing.corr()
# housing_copy.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

# fix missing values for attribute total_bedrooms
# to do this we pad up missing values with the median of values for the attribute
# This median value needs to be stored for the current dataset and future data sets.
# Scikit-learn provides a class to do this (Imputer)
imputer = Imputer(strategy='median')

# As the median can only be computed against numerical attributes. We need a copy of the
# dataset without the attribute 'ocean_proximity' which is not numeric
housing_num = housing_copy.drop('ocean_proximity', axis=1)

# Fit the Imputer instance to the training data using the fit() method
imputer.fit(housing_num)

# Uncomment below to check trained imputer statistics
# imputer.statistics_

# The Imputer computes the median for each attribute and stores the result
# in its statistics_instance variable.
# We apply to all numeric attributes to make sure we fill all attributes that may have empty values.
# Use this 'trained' imputer to transform the training seet by
# replacing missing values by the learned medians
x = imputer.transform(housing_num)

# Handle text attributes like Categorical attribute 'ocean_proximity'
# Encode the numerical values
housing_cat = housing_copy['ocean_proximity']
# Use LabelBinarizer to transform from  text category -> integer category -> one-hot vectors (dense)

encoder_lbrz = LabelBinarizer()
housing_cat_1hot = encoder_lbrz.fit_transform(housing_cat)

# Customr transformation
# Using our CombinedAttributesAdder -> which at the moment has been gated on 'add_bedrooms_per_room' hyperparameter
# to False
attr_adder = Ct.CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing_copy.values)

# Feature Scaling
# Min-max (normalization) scaling. Values are shifted and rescaled so that they in the range 0-1
#   How: Subtracting the min value and dividing by the max minus the min. Use Sklearn 'MinMaxScaler
#   Can be affected by outliers - as it crushes all values within the set.
# Standardization: Subtracts the mean value divide by the variance so that the resulting distribution has unit variance.
#   Standardization does not bound values to a specific range. (May be problem for some Algos - neural networks
#   often expect an input value ranging from 0 -1).
#   Standardization is much less affected by outliers.


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
num_pipeline = Pipeline([('selector', Cs.DataFrameSelector(num_attribs)),
                         ('imputer', Imputer(strategy='median')),
                         ('attribs_adder', Ct.CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])

cat_pipeline = Pipeline([('selector', Cs.DataFrameSelector(cat_attribs)),
                         ('lable_binarizer', LabelBinarizer())])

full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline), ('cat_pipeline', cat_pipeline)])
housing_prepared = full_pipeline.fit_transform(housing_copy)
print('Finished housing Prepared')
print(housing_prepared)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_copy_labels)
print('Finished fitting')

# let's try it out on few instances from the training set
some_data = housing_copy.iloc[:5]
some_labels = housing_copy_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:\t\t', lin_reg.predict(some_data_prepared))
print('Labels:\t\t', list(some_labels))

# let's measure
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_copy_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_copy_labels)
tree_housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_copy_labels, tree_housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_copy_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_copy_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(tree_rmse_scores)
display_scores(lin_rmse_scores)

# RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_copy_labels)
forest_housing_predictions = forest_reg.predict(housing_prepared)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_copy_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



