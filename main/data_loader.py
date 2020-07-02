import json
import ast
from main.constants import *

def load_features(
	input_file,
	load_mapping=False,
	key_val_separator=KEY_VAL_SEPARATOR
	):
	"""
	input_file: input_file_path
		each line is item in feature list
	load_mapping: whether or not to load (feature, value) mapping
	key_val_separator: separator between feature and value if load_mapping

	output:
		list / dictionary of features
	"""
	if load_mapping:
		features = dict()
		try:
			with open(input_file, "r") as file:
				lines = file.readlines()
				for line in lines:
					[feature_name, feature_value] = line.strip('\n').split(key_val_separator)
					if feature_name not in features:
						features[feature_name] = list()
					features[feature_name].append(feature_value)

		except IOError:
			print("{} doesn't exists".format(input_file))

	else:
		features = list()
		try:
			with open(input_file, "r") as file:
				lines = file.readlines()
				for line in lines:
					features.append(line)

		except IOError:
			print("{} doesn't exists".format(input_file))

	return features


def load_feature_vec_to_int(input_file):
	"""
	input_file: input_file_path
		each line is json object contains feature_id and feature_dict

	output:
		mapping from feature_dict to feature_id
	"""
	feature_vector_to_int = dict()
	try:
		with open(input_file, "r") as file:
			lines = file.readlines()
			for line in lines:
				line = json.load(line.strip('\n'))
				feature_dict = line[FEATURE_DICT]
				feature_id = line[FEATURE_ID]
				feature_vector_to_int[frozenset(feature_dict.items())] = feature_id

	except IOError:
		print("{} doesn't exists".format(input_file))

	return feature_vector_to_int

def load_mapping(
	input_file,
	value_is_set,
	eval_set={0,1},
	col_separator=COL_SEPARATOR
	):
	"""
	input_file: input_file_path
		each line is (key, value) pair separated by col_separator
	value_is_set: whether or not value is set
	eval_set: for each line of (key, value), indices to calculate
	col_separator: separator between key and values

	output:
		mapping from key to value in dictionary
	"""
	mapping = dict()
	try:
		with open(input_file, "r") as file:
			lines = file.readlines()
			for line in lines:
				[key, val] = line.strip('\n').split(col_separator)

				if 0 in eval_set:
					key = ast.literal_eval(key)
				
				if 1 in eval_set:
					val = ast.literal_eval(val)

				if value_is_set:
					if key not in mapping:
						mapping[key] = list()
					mapping[key].append(val)

				else:
					mapping[key] = val

	except IOError:
		print("{} doesn't exists".format(input_file))

	return mapping

def load_reviews(
	input_file,
	eval_set={0,1,2,3},
	col_separator=COL_SEPARATOR
	):
	"""
	input_file: input_file_path
		each line is col_separateor separated review [int_user_id, restaurant, rating, timestamp]
	eval_set: for each review, indices to calculate
	col_separator: separator between key and values

	output:
		reviews: list of reviews [int_user_id, restaurant, rating, timestamp]
	"""
	reviews = list()
	try:
		with open(input_file, "r") as file:
			lines = file.readlines()
			for line in lines:
				[user, restaurant, rating, timestamp] = line.strip('\n').split(col_separator)
				if 0 in eval_set:
					user = ast.literal_eval(user)

				if 1 in eval_set:
					restaurant = ast.literal_eval(restaurant)
				
				if 2 in eval_set:
					rating = ast.literal_eval(rating)

				if 3 in eval_set:
					timestamp = ast.literal_eval(timestamp)

				reviews.append([user, restaurant, rating, timestamp])

	except IOError:
		print("{} doesn't exists".format(input_file))

	return reviews