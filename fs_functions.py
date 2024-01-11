# imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import itertools
import math
import statistics
from itertools import permutations
import json
import time
import joblib
from scipy.stats import wasserstein_distance
# from pyemd import emd
import sys
import matplotlib.pyplot as plt

from itertools import permutations


# We only take numerical columns to plot on the axis
def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False

def FS1(df, group_label, group_label_values):
	# Data cleaning, cars dataset contains "?" as the missing value.
	# All other datasets in the paper do not have a missing values

	# print(f"Group label is: {group_label}")
	# print(f"Group label has {group_label_values.nunique()} classes")
	size_of_dataset = df.shape[0]

	# We take only numerical columns and delete the others.
	colu = []
	for c in df.columns:
		if (isfloat(df[c].iloc[0]) and c != group_label):
			colu.append(c)

	df_temp = df[colu]

	# First step is the normalization of the columns (range of values is [0,1])
	scaler = MinMaxScaler()
	df_temp[df_temp.columns] = scaler.fit_transform(df_temp[df_temp.columns])

	df_temp[group_label] = group_label_values
	df = df_temp.copy()

	# Calculating mean values for each column and each subset
	n_classes = df[group_label].unique()
	dict_of_mean_values = {}  # key is the origin and the values are list of means for each column of a datas

	# Splitting the data into the subsets and then calculating the mean for each column
	for group in n_classes:
		df_group = df[df[group_label] == group].copy()

		df_group.drop(group_label, axis=1, inplace=True)
		dict_of_mean_values[group] = [df_group[c].mean() for c in df_group.columns]

	# Calculating differences for columns between each combination of classes
	diff = []
	weights = []
	for key1, key2 in itertools.combinations(dict_of_mean_values.keys(), 2):
		diff.append(abs(np.subtract(np.array(dict_of_mean_values[key1]), np.array(dict_of_mean_values[key2]))))
		weight = (df[df[group_label] == key1].shape[0] + df[df[group_label] == key2].shape[0]) / (
					size_of_dataset * (len(n_classes) - 1))
		weights.append(weight)

	df.drop(group_label, axis=1, inplace=True)
	# we dont need group label anymore, from this point further we
	# are calculating QDS and TDS and fo that we do not need group label column

	D = df.shape[1]
	W = np.ones((D, D))
	for row in range(len(W)):
		for col in range(len(W[0])):
			dij = abs(row - col)
			W[row][col] = 1 - (dij / (D - 1))

	QFD = []
	for count, d in enumerate(diff):
		QFD.append(math.sqrt(np.dot(np.dot(d, W), np.transpose(d))))

	tds = 0
	for i in range(len(QFD)):
		tds += QFD[i] * weights[i]

	return tds


from itertools import permutations


# We only take numerical columns to plot on the axis
def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False


def FS2(df, group_label, group_label_values):
	# Data cleaning, cars dataset contains "?" as the missing value.
	# All other datasets in the paper do not have a missing values
	# print(f"Group label is: {group_label}")
	# print(f"Group label has {group_label_values.nunique()} classes")
	size_of_dataset = df.shape[0]

	# We take only numerical columns and delete the others.
	colu = []
	for c in df.columns:
		if (isfloat(df[c].iloc[0]) and c != group_label):
			colu.append(c)

	df_temp = df[colu]

	# First step is the normalization of the columns (range of values is [0,1])
	scaler = MinMaxScaler()
	df_temp[df_temp.columns] = scaler.fit_transform(df_temp[df_temp.columns])

	df_temp[group_label] = group_label_values
	df = df_temp.copy()

	# Calculating mean and standard deviation for each column and each subset
	n_classes = df[group_label].unique()
	dict_of_mean_values = {}  # key is the origin and the values are list of means for each column of a datas
	dict_of_std_values_minus = {}
	dict_of_std_values_plus = {}
	final_dict = {}

	# Splitting the data into the subsets and then calculating the mean for each column
	for group in n_classes:
		df_group = df[df[group_label] == group].copy()

		df_group.drop(group_label, axis=1, inplace=True)

		dict_of_std_values_minus[group] = [df_group[c].mean() - df_group[c].std() for c in df_group.columns]

		dict_of_mean_values[group] = [df_group[c].mean() for c in df_group.columns]

		dict_of_std_values_plus[group] = [df_group[c].mean() + df_group[c].std() for c in df_group.columns]

		# We are creating a matrix of these comupted values (the shape is 3xNumberOfColumnsThatRemained)
		matrix = np.array([dict_of_std_values_minus[group], dict_of_mean_values[group], dict_of_std_values_plus[group]])
		final_dict[group] = matrix
	# Calculating differences for columns between each combination of classes
	QFD = []

	D = df.shape[1] - 1  # we do not count group label as the dimension in D, This represents number of dimensions
	W = np.ones((D, D))
	for row in range(len(W)):
		for col in range(len(W[0])):
			dij = abs(row - col)
			W[row][col] = 1 - (dij / (D - 1))

	first_key = next(iter(final_dict.keys()))
	no_dimensions = final_dict[first_key].shape[1]  # number of columns in a matrix which is number of dimensions
	weights = []

	for key1, key2 in itertools.combinations(final_dict.keys(), 2):
		diff = []

		# Calculating difference between two groups for -std, maean and +std
		for i in range(0, no_dimensions):
			# We got thorugh each dimension, this mean that we are taking column by column and calculating Earth movers distance
			# between each column respectively.
			column1 = final_dict[key1][:, i]
			column2 = final_dict[key2][:, i]
			wasserstein_d = wasserstein_distance(column1, column2)
			diff.append(wasserstein_d)

		weight = (df[df[group_label] == key1].shape[0] + df[df[group_label] == key2].shape[0]) / (
					size_of_dataset * (len(n_classes) - 1))
		weights.append(weight)
		QFD.append(math.sqrt(np.dot(np.dot(diff, W), np.transpose(diff))))

	# In the paper the following is stated for FS2 and FS3: "FS2 (mean and standard deviation) and
	# FS3 (histogram) are multidimensional vectors, where each dimension consists of the mean and variance measure, or bins of the histogram. For these, we compute a difference value for each dimension as the Earth Mover Distance"

	tds = 0
	for i in range(len(QFD)):
		tds += QFD[i] * weights[i]

	return tds


from itertools import permutations


# We only take numerical columns to plot on the axis
def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False


def FS3(df, group_label, group_label_values):
	# Data cleaning, cars dataset contains "?" as the missing value.
	# All other datasets in the paper do not have a missing values
	# print(f"Group label is: {group_label}")
	# print(f"Group label has {group_label_values.nunique()} classes")
	size_of_dataset = df.shape[0]

	# We take only numerical columns and delete the others.
	colu = []
	for c in df.columns:
		if (isfloat(df[c].iloc[0]) and c != group_label):
			colu.append(c)

	df_temp = df[colu]

	# First step is the normalization of the columns (range of values is [0,1])
	scaler = MinMaxScaler()
	df_temp[df_temp.columns] = scaler.fit_transform(df_temp[df_temp.columns])

	df_temp[group_label] = group_label_values
	df = df_temp.copy()

	# Calculating 5 bin histogram for each column and each subset
	n_classes = df[group_label].unique()

	final_dict = {}

	# Splitting the data into the subsets and then calculating the Histogram with 5 bins for each column
	for group in n_classes:

		df_group = df[df[group_label] == group].copy()

		df_group.drop(group_label, axis=1, inplace=True)

		hist_matrix = np.zeros((5, len(df_group.columns)))

		for i, column in enumerate(df_group.columns):
			hist_values, _ = np.histogram(df_group[column], bins=5)

			# hist_matrix[:, i] = hist_values
			# Normalization of the histogram values
			total_points = len(df_group[column])
			normalized_hist_values = hist_values / total_points
			hist_matrix[:, i] = normalized_hist_values

		# We are creating a matrix of these comupted values (the shape is 5xNumberOfColumnsThatRemained)
		final_dict[group] = hist_matrix
	# Calculating differences for columns between each combination of classes
	QFD = []

	D = df.shape[1] - 1  # we do not count group label as the dimension in D, This represents number of dimensions
	W = np.ones((D, D))
	for row in range(len(W)):
		for col in range(len(W[0])):
			dij = abs(row - col)
			W[row][col] = 1 - (dij / (D - 1))

	first_key = next(iter(final_dict.keys()))
	no_dimensions = final_dict[first_key].shape[1]  # number of columns in a matrix which is number of dimensions
	weights = []

	for key1, key2 in itertools.combinations(final_dict.keys(), 2):
		diff = []

		# Calculating difference between two groups for -std, maean and +std
		for i in range(0, no_dimensions):
			# We got thorugh each dimension, this mean that we are taking column by column and calculating Earth movers distance
			# between each column respectively.
			column1 = final_dict[key1][:, i]
			column2 = final_dict[key2][:, i]
			wasserstein_d = wasserstein_distance(column1, column2)
			diff.append(wasserstein_d)

		weight = (df[df[group_label] == key1].shape[0] + df[df[group_label] == key2].shape[0]) / (
					size_of_dataset * (len(n_classes) - 1))
		weights.append(weight)
		QFD.append(math.sqrt(np.dot(np.dot(diff, W), np.transpose(diff))))

	# In the paper the following is stated for FS2 and FS3: "FS2 (mean and standard deviation) and
	# FS3 (histogram) are multidimensional vectors, where each dimension consists of the mean and variance measure, or bins of the histogram. For these, we compute a difference value for each dimension as the Earth Mover Distance"

	tds = 0
	for i in range(len(QFD)):
		tds += QFD[i] * weights[i]

	return tds


def run_permutations(df, group_label, file_path, func):
	df = df[df.apply(lambda row: '?' not in row.values, axis=1)].copy()
	colu = []
	group_label_values = df[group_label]
	for c in df.columns:
		if (isfloat(df[c].iloc[0]) and c != group_label):
			colu.append(c)

	df = df[colu]
	col_keys = [i for i in range(0, len(df.columns.tolist()))]
	tds_list = []
	permutations_list = list(permutations(col_keys))
	permutations_list = [list(tup) for tup in permutations_list]

	permutations_list_names = permutations_list.copy()
	og_order = df.columns.tolist()

	# Create a list of all permutations.
	for i in range(len(permutations_list)):
		permutations_list_names[i] = [og_order[index] for index in permutations_list[i]]

	df[group_label] = group_label_values
	for i in range(len(permutations_list)):
		subset = df.iloc[:, permutations_list[i]]

		tds_list.append(func(subset, group_label,
							 group_label_values))  # subset must have group label column in the dataframe. But we should not include this column when calculating all possible permutations.

	# the resulting dictionary along with the actual column names
	result_cars_cols = {
		index: {"columns_order": permutations_list_names[index], "tds": tds_list[index]}
		for index, _ in enumerate(permutations_list_names)
	}

	# the resulting dictionary along with the indexes of the column names as was ordered in the original ordering
	result_cars_indexes = {
		index: {"columns_order": permutations_list[index], "tds": tds_list[index]}
		for index, _ in enumerate(permutations_list)
	}

	with open(file_path, "w") as json_file:
		json.dump(result_cars_cols, json_file)