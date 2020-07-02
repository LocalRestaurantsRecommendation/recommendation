import os
import pandas as pd
import numpy as np
from datetime import datetime

from main.constants import *
from main.data_loader import *
from main.recommendar import *

def one_for_all_load():
	"""
	helper function
	load reviews, user_city, rest_city
	"""
	reviews = load_reviews(DATA_DIR + "dedup_filtered_reviews.data")
	
	user_city = load_mapping(
		input_file=DATA_DIR + "int_user_id_to_cities.data", 
		value_is_set=True,
		eval_set={0}
		)
	rest_city = load_mapping(
		input_file=DATA_DIR + "feature_id_to_cities.data", 
		value_is_set=False,
		eval_set={0}
		)

	rest_id_to_int = load_mapping(
		input_file=DATA_DIR + "restaurants_id_to_int.data", 
		value_is_set=False,
		eval_set={1}
		)
	return reviews, user_city, rest_city, rest_id_to_int

def evaluate_models_for_rounds(
	rounds=10,
	output_dir=OUTPUT_DIR,
	models=["baseline"],
	evaluate_on_sample_users=True,
	sample_list=None,
	sample_limit=70,
	seed=None,
	min_rating=50,
	k=TOP_K,
	removeSeen=True,
	infer_loc_by_latest_rating_only=True,
	latest_rating_limiter=3,
	training_number_params=[0.3, 0.5, 0.7],
	is_params_ratio=True,
	col_separator=COL_SEPARATOR
	):
	"""
	Do rounds of evaluate_models_on_yelp_open_dataset

		rounds:
			rounds to do evaluate_models_on_yelp_open_dataset

		output_dir:
			place to hold output files

		models:
			list of model to evaluate on

		evaluate_on_sample_users:
			true, we evaluate on sample users
				either with sample list or sample limit

			false, evaluate on all users with >= min_rating

		sample_list:
			list of users to evaluate on

		sample_limit:
			maximum number of sample users to evaluate on

		seed:
			list of seeds for deterministic sampling
			if length is not same as rounds, append with None

		min_rating:
			filter out pool of users to do sample
			we only evaluate on sample users with >= min_rating

		k:
			parameter for Recommendar
			top k items to recommend

		removeSeen:
			true, we don't recommend items users rated
			false, otherwise

		infer_loc_by_latest_rating_only:
			infer user's location by latest rating or all ratings

		latest_rating_limiter:
			hyperparameters for whole recommendar
			number of latest rating to consider for each user

		training_number_params:
			hyperparameters for each user
			can be list of number or ratios

			number n, meaning we know user's first n ratings
			ratio n, meaning we know users first n% ratings

			here small numbers make more sense in reality

		is_params_ratio:
			whether or not training_number_params is list of ratio

		col_separator:
			separator for output file	
	"""
	filedir = "{}{}".format(
			output_dir, 
			datetime.today().strftime('%Y-%m-%d-%H:%M:%S/')
			)
	os.makedirs(os.path.dirname(filedir), exist_ok=True)

	if seed is None:
		seed = [ None for x in range(rounds) ]

	if len(seed) < rounds:
		seed += [ None ] * (rounds - len(seed))

	for i in range(rounds):
		evaluate_models_on_yelp_open_dataset(
			output_dir=filedir,
			models=models,
			evaluate_on_sample_users=evaluate_on_sample_users,
			sample_list=sample_list,
			sample_limit=sample_limit,
			seed=seed[i],
			min_rating=min_rating,
			k=k,
			removeSeen=removeSeen,
			infer_loc_by_latest_rating_only=infer_loc_by_latest_rating_only,
			latest_rating_limiter=latest_rating_limiter,
			training_number_params=training_number_params,
			is_params_ratio=is_params_ratio,
			col_separator=col_separator
			)

def evaluate_models_on_yelp_open_dataset(
	output_dir=OUTPUT_DIR,
	models=["baseline"],
	evaluate_on_sample_users=True,
	sample_list=None,
	sample_limit=70,
	seed=None,
	min_rating=100,
	k=TOP_K,
	removeSeen=True,
	infer_loc_by_latest_rating_only=True,
	latest_rating_limiter=3,
	training_number_params=[1,3,5,7,10],
	is_params_ratio=False,
	col_separator=COL_SEPARATOR
	):
	"""
	evaluate models on yelp open dataset by sample
	* sampling becuase of memory limit on local machine *

		output_dir:
			place to hold output files

		models:
			list of model to evaluate on

		evaluate_on_sample_users:
			true, we evaluate on sample users
				either with sample list or sample limit

			false, evaluate on all users with >= min_rating

		sample_list:
			list of users to evaluate on

		sample_limit:
			maximum number of sample users to evaluate on

		seed:
			for deterministic sampling

		min_rating:
			filter out pool of users to do sample
			we only evaluate on sample users with >= min_rating

		k:
			parameter for Recommendar
			top k items to recommend

		removeSeen:
			true, we don't recommend items users rated
			false, otherwise

		infer_loc_by_latest_rating_only:
			infer user's location by latest rating or all ratings

		latest_rating_limiter:
			hyperparameters for whole recommendar
			number of latest rating to consider for each user

		training_number_params:
			hyperparameters for each user
			can be list of number or ratios

			number n, meaning we know user's first n ratings
			ratio n, meaning we know users first n% ratings

			here small numbers make more sense in reality

		is_params_ratio:
			whether or not training_number_params is list of ratio

		col_separator:
			separator for output file

	output file:
		report.data
	"""
	reviews, user_city, rest_city, rest_id_to_int = one_for_all_load()
	user_ratings = map_user_to_ratings(reviews)

	user_list_all = list()
	for (user, ratings) in user_ratings.items():
		if ratings >= min_rating:
			user_list_all.append(user)

	if seed is not None:
		np.random.seed(seed)

	if evaluate_on_sample_users:
		if sample_list is None:
			if len(user_list_all) > sample_limit:
				user_list = list(np.random.choice(user_list_all, size=sample_limit))
			else:
				user_list = user_list_all
		else:
			user_list = sample_list
	else:
		user_list = user_list_all

	filedir = "{}{}".format(
			output_dir, 
			datetime.today().strftime('%Y-%m-%d-%H:%M:%S/')
			)
	os.makedirs(os.path.dirname(filedir), exist_ok=True)

	filename = filedir + "report.data"
	with open(filename, "a") as file:
		file.write(f"models = {models}\n")
		file.write(f"evaluate_on_sample_users = {evaluate_on_sample_users}\n")
		file.write(f"sample_list = {sample_list}\n")
		file.write(f"sample_limit = {sample_limit}\n")
		file.write(f"seed = {seed}\n")
		file.write(f"actual seed = {np.random.get_state()[2]}\n")
		file.write(f"min_rating = {min_rating}\n")
		file.write(f"k = {k}\n")
		file.write(f"removeSeen = {removeSeen}\n")
		file.write(f"infer_loc_by_latest_rating_only = {infer_loc_by_latest_rating_only}\n")
		file.write(f"latest_rating_limiter = {latest_rating_limiter}\n")
		file.write(f"training_number_params = {training_number_params}\n")
		file.write(f"is_params_ratio = {is_params_ratio}\n")
		file.write(f"col_separator = {col_separator}\n")
		file.write(f"user_list = {user_list}\n")

	for model in models:
		recommendar = get_recommendar(
			reviews=reviews,
			user_city=user_city,
			rest_city=rest_city,
			rest_id_to_int=rest_id_to_int,
			model=model, 
			k=k, 
			removeSeen=removeSeen,
			infer_loc_by_latest_rating_only=infer_loc_by_latest_rating_only,
			latest_rating_limiter=latest_rating_limiter
			)

		apk_all = recommendar.train_for_all_user(
			output_dir=filedir,
			user_list=user_list,
			train_number_params=training_number_params,
			is_params_ratio=is_params_ratio,
			col_separator=col_separator
			)

		with open(filename, "a") as file:
			file.write(f"{model}{col_separator}{apk_all}\n")