import os
import pandas as pd
import numpy as np
from datetime import datetime

from main.constants import *
from main.data_loader import *
from main.evaluation import *
from main.model.baseline_model import Model as BaselineModel
from main.model.als_model import Model as AlsModel
from main.model.naive_hybrid_baseline_als_model import Model as NHBAModel
from main.model.time_biased_hybrid_model import Model as TBHModel

#-------------------------------
# Constants
#-------------------------------
MAPPING_MODEL = {
	"baseline" : BaselineModel,
	"als" : AlsModel,
	"nhba" : NHBAModel,
	"tbh" : TBHModel
}

#-------------------------------
# Helper Functions
#-------------------------------
def map_user_to_ratings(reviews):
	"""
	helper function
	return mapping from user_id to num of ratings in reviews
	"""
	user_ratings = dict()
	for review in reviews:
		user = review[0]
		if user not in user_ratings:
			user_ratings[user] = 0
		else:
			user_ratings[user] = user_ratings[user] + 1

	return user_ratings

def map_feature_ids_to_restaurants(feature_list, rest_id_to_int):
	"""
	helper function
	return list of restaurants mapped from feature in feature_list
	in same order as feature_list
	"""

	invert_mapping = dict()
	for (rest_id, feature) in rest_id_to_int.items():
		if feature not in feature_list:
			continue

		if feature not in invert_mapping:
			invert_mapping[feature] = list()

		invert_mapping[feature].append(rest_id)

	rest_list = list()
	for feature in feature_list:
		rest_list += invert_mapping[feature]

	return rest_list

#-------------------------------
# APIs
#-------------------------------
def get_recommendar(
	reviews,
	user_city,
	rest_city,
	rest_id_to_int,
	model="baseline",
	k=TOP_K,
	removeSeen=True,
	infer_loc_by_latest_rating_only=True,
	latest_rating_limiter=3
	):
	"""
	factory method to get a recommendar instance

		reviews:
			list of reviews to consider, should contain all reviews updated
			filename to load reviews data

		user_city:
			mapping from int_user_id to city
			filename to load user_city data

		rest_city:
			mapping from feature_id to city
			filename to load rest_city data

		rest_id_to_int:
			mapping from restaurants id to feature id
			filename to load restaurants id to feature id

		model:
			model for train

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
	"""
	return LocalRecommendar(
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

class LocalRecommendar:
	"""
	state:
		self.reviews : list of reviews from review_file
		self.user_city : int_user_id maps to cities
		self.rest_city : feature_id maps to cities
		self.rest_id_to_int : restaurant_id maps to feature_id
		self.model : model to do recommendation
		self.k : top k restaurants' feature id to recommend
		self.removeSeen : whether or not recommend seen feature_id
		self.reviews_dataframe : pandas dataframe of reviews
		self.user_ratings : mapping from user id to num of ratings
		self.infer_loc_by_latest_rating_only : whether or not consider latest rating
		self.latest_rating_limiter : num of latest rating to consider
	"""
	def __init__(
		self,
		reviews,
		user_city,
		rest_city,
		rest_id_to_int,
		model="baseline",
		k=TOP_K,
		removeSeen=True,
		infer_loc_by_latest_rating_only=True,
		latest_rating_limiter=3
		):
		"""
		initialize all state

			reviews:
				list of reviews to consider, should contain all reviews updated
				filename to load reviews data

			user_city:
				mapping from int_user_id to city
				filename to load user_city data

			rest_city:
				mapping from feature_id to city
				filename to load rest_city data

			rest_id_to_int:
				mapping from restaurants id to feature id
				filename to load restaurants id to feature id

			model: 
				string name for model

			k: 
				top k restaurants' feature id to recommend

			removeSeen: 
				whether or not recommend seen feature_id

			infer_loc_by_latest_rating_only:
				infer user's location by latest rating or all ratings

			latest_rating_limiter:
				hyperparameters for whole recommendar
				number of latest rating to consider for each user
		"""
		
		# load reviews
		if isinstance(reviews, list):
			self.reviews = reviews
		elif isinstance(reviews, str):
			self.reviews = load_reviews(reviews)
		else:
			raise ValueError(f"reviews = {reviews} has invalid type: not list nor str")

		# load user_city
		if isinstance(user_city, dict):
			self.user_city = user_city
		elif isinstance(user_city, str):
			self.user_city = load_mapping(
				input_file=user_city, 
				value_is_set=True,
				eval_set={0}
				)
		else:
			raise ValueError(f"user_city = {user_city} has invalid type: not dict nor str")

		# load rest_city
		if isinstance(rest_city, dict):
			self.rest_city = rest_city
		elif isinstance(rest_city, str):
			self.rest_city = load_mapping(
				input_file=rest_city, 
				value_is_set=False,
				eval_set={0}
				)
		else:
			raise ValueError(f"rest_city = {rest_city} has invalid type: not dict nor str")

		# load rest_id_to_int
		if isinstance(rest_id_to_int, dict):
			self.rest_id_to_int = rest_id_to_int
		elif isinstance(rest_id_to_int, str):
			self.rest_id_to_int = load_mapping(
				input_file=rest_id_to_int, 
				value_is_set=False,
				eval_set={1}
				)
		else:
			raise ValueError(f"rest_id_to_int = {rest_id_to_int} has invalid type: not dict nor str")

		# load model
		if model not in MAPPING_MODEL:
			raise ValueError(f"model {model} doesn't exist")

		self.model = model
		self.k = k
		self.removeSeen = removeSeen
		self.infer_loc_by_latest_rating_only = infer_loc_by_latest_rating_only
		self.latest_rating_limiter = latest_rating_limiter


		self.reviews_dataframe = pd.DataFrame(
			self.reviews, 
			columns=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP]
			)

		self.user_ratings = map_user_to_ratings(self.reviews)

	def train_for_all_user(
		self,
		output_dir,
		user_list,
		train_number_params,
		is_params_ratio=False,
		col_separator=COL_SEPARATOR
		):
		"""
		train for all users in user_list

			output_dir: 
				place to put output files

			user_list: 
				list of users to train on

			training_number_params:
				hyperparameters for each user
				can be list of number or ratios

				number n, meaning we know user's first n ratings
				ratio n, meaning we know users first n% ratings

				here small numbers make more sense in reality

			is_params_ratio:
				whether or not training_number_params is list of ratio

		output files:
		user_rating_info.data:
			each line contains 
			int_user_id, total_ratings, best_train_number, average precision @ k, precision @ k, recall @ k
			1	300	10	0.9	0.5	0.5

		return mean average precision @ k over user_list
		"""
		print("Woking on {} users...".format(len(user_list)))

		best_apk = list()
		best_pk = list()
		best_rk = list()
		best_tn = list()

		filename = "{}{}".format(output_dir, f"{self.model}_user_rating_info.data")
		for i in range(len(user_list)):
			user = user_list[i]
			train_number = train_number_params

			if is_params_ratio:
				train_number = [int(x * self.user_ratings[user]) for x in train_number_params]

			pk, rk, apk = self.personalized_train(
				user_id=user,
				train_number=train_number
				)

			best_index = np.argmax(apk)
			best_tn.append(train_number[best_index])
			best_apk.append(apk[best_index])
			best_pk.append(pk[best_index])
			best_rk.append(rk[best_index])

			with open(filename, "a") as file:
				file.write("{}{}{}{}{}{}{}{}{}{}{}\n".format(
					user, col_separator,
					self.user_ratings[user], col_separator,
					best_tn[i], col_separator,
					best_apk[i], col_separator,
					best_pk[i], col_separator,
					best_rk[i]
					))

		return np.sum(best_apk) / len(user_list)

	def personalized_train(
		self,
		user_id,
		train_number=[1, 3, 5]
		):
		"""
		do training on single user

			user_id: 
				integer id for a user

			train_number: 
				list of number of ratings to consider for this user

		return: 
			list of precision @ k
			list of recall @ k
			list of average precision @ k
			for each train number
		"""
		pk_list = list()
		rk_list = list()
		apk_list = list()

		for number in train_number:
			model_name="{}_{}_{}".format(self.model, user_id, number)

			print("\nWorking on {}".format(model_name))
			print("\tLoad last_timestamp ...")
			first_number_rating = self.reviews_dataframe[self.reviews_dataframe[COL_USER] == user_id] \
										.sort_values(COL_TIMESTAMP) \
										.head(number)
			last_timestamp = int(first_number_rating.iloc[number - 1][COL_TIMESTAMP])
			print("\tlast_timestamp = {}".format(last_timestamp))

			# only consider rated cities
			city_list_all = list()
			for index, row in first_number_rating.iterrows():
				city_list_all.append(self.rest_city[row[COL_ITEM]])

			# only consider cities for limited number of latest rating
			if self.infer_loc_by_latest_rating_only:
				city_list = set()
				index = len(city_list_all) - 1
				while index >= 0 and index + self.latest_rating_limiter >= len(city_list_all):
					city_list.add(city_list_all[index])
					index -= 1
			else:
				city_list = set(city_list_all)

			# only consider restaurants in same cities
			restaurant_list = list()
			for (restaurant, city) in self.rest_city.items():
				if city in city_list:
					restaurant_list.append(restaurant)

			print("\tLoad subset_reviews ...")
			subset_reviews = self.reviews_dataframe[
				(self.reviews_dataframe[COL_ITEM].isin(restaurant_list)) 
				& (self.reviews_dataframe[COL_TIMESTAMP] <= last_timestamp)
				]
			print("\tsubset_reviews: {}".format(subset_reviews.shape))

			print("\tLoad model ...")
			model = self._get_model(
				user_id=user_id,
				model_name=model_name,
				data_reviews=subset_reviews
				)

			print("\tModel trained ...")
			model.train()

			print("\tModel predicted ...")
			res_dataframe = model.predict(
				k=self.k,
				removeSeen=self.removeSeen
				)

			model.close()

			print("\tres_dataframe: {}".format(res_dataframe.shape))
			print("\tCalculate APK & PK & RK ...")
			pk, rk, apk = evaluate_top_k_for_user(
				self.reviews_dataframe[
					(self.reviews_dataframe[COL_USER] == user_id) 
					& (self.reviews_dataframe[COL_TIMESTAMP] > last_timestamp)
					],
				res_dataframe
				)

			pk_list.append(pk)
			rk_list.append(rk)
			apk_list.append(apk)

		print("\tpk_list: {}".format(pk_list))
		print("\trk_list: {}".format(rk_list))
		print("\tapk_list: {}".format(apk_list))

		return pk_list, rk_list, apk_list

	def predict(self, user_id):
		"""
		use all user's rating behavior up to now to predict future

		user_id: integer id for a user

		return:
			list of top k feature_ids
		"""

		# only consider cities for limited number of latest rating
		if self.infer_loc_by_latest_rating_only:
			first_number_rating = self.reviews_dataframe[self.reviews_dataframe[COL_USER] == user_id] \
									.sort_values(COL_TIMESTAMP, ascending=False)

			city_list = set()
			index = 0
			while index < first_number_rating.shape[0] and index < self.latest_rating_limiter:
				city_list.add(self.rest_city[first_number_rating.iloc[index][COL_ITEM]])
				index += 1
		else:
			city_list = set(city_list_all)

		# only consider restaurants in same cities
		restaurant_list = list()
		for (restaurant, city) in self.rest_city.items():
			if city in city_list:
				restaurant_list.append(restaurant)

		subset_reviews = self.reviews_dataframe[
				self.reviews_dataframe[COL_ITEM].isin(restaurant_list)
				]

		model = self._get_model(
				user_id=user_id,
				model_name="{}_{}".format(self.model, user_id),
				data_reviews=subset_reviews
				)

		model.train()

		res_dataframe = model.predict(
				k=self.k,
				removeSeen=self.removeSeen
				)

		model.close()

		return map_feature_ids_to_restaurants(
			feature_list=res_dataframe[COL_ITEM].tolist(),
			rest_id_to_int=self.rest_id_to_int
			)

	def _get_model(self, user_id, model_name, data_reviews):
		"""
		factory method to get user personalized model
		"""
		return MAPPING_MODEL[self.model](
			user_id=user_id,
			model_name=model_name,
			data_reviews=data_reviews
			)