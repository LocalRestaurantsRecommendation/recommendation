import os
import json
import ast
from datetime import datetime
from main.constants import *

#-------------------------------
# Constants
#-------------------------------
# business
HOURS = "hours"
CATEGORIES = "categories"
ATTRIBUTES = "attributes"
CITY = "city"

WORKDAYS = "workdays"
WORKDAYS_VALUE = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORKDAYS_START = lambda day : "{}_start_time".format(day)
WORDDAYS_END = lambda day : "{}_end_time".format(day)

# reviews
RATING = "stars"
TIMESTAMP = "date"
USER_ID = "user_id"
RESTAURANT_ID = "business_id"

#-------------------------------
# Helper Functions
#-------------------------------
def is_open(business):
	"""
	business: json object of business

	check business is open or not
	"""
	return business["is_open"] == 1

def is_restaurant(business):
	"""
	business: json object of business

	check business is restaurants or not
	"""
	return "Restaurants" in business[CATEGORIES]

def is_valid_business(business):
	"""
	business: json object of business

	check business is valid business or not

	valid means business has "attributes", "categories", "city" & "hours"
	"""
	return business[ATTRIBUTES] is not None \
			and business[CATEGORIES] is not None \
			and business[CITY] is not None \
			and business[HOURS] is not None

def preprocess_categories(
	business,
	exclude_words=["Food", "Restaurants"]
	):
	"""
	modify business json object
	change original categories to a list of keywords
	delete keywords some of keywords

	business: json object of business
	exclude_words: list of words to exclude in categories
	"""
	line = business[CATEGORIES].split(", ")
	cate_list = [x for x in line if x not in exclude_words]
	business[CATEGORIES] = cate_list

def preprocess_hours_extend_workdays(business):
	"""
	modify business json object
	add a "workdays" attribute, which maps to a list of workdays
	"""
	workdays = list()
	for (day, hour) in business[HOURS].items():
		workdays.append(day)
		start_end = hour.split("-")
		business[WORKDAYS_START(day)] = start_end[0]
		business[WORDDAYS_END(day)] = start_end[1]

	business[WORKDAYS] = workdays

def preprocess_attributes(business):
	"""
	modify business json object
	"attributes" should be a clean mapping without None value
	"""
	attrs = business[ATTRIBUTES]
	attrs_dict = dict()

	for (key, val) in attrs.items():
		val = ast.literal_eval(val)
		if val is None:
			continue

		# if val is a json object
		if isinstance(val, dict):
			inner_dict = dict()
			for (inner_key, inner_val) in val.items():
				inner_key = inner_key
				inner_val = inner_val
				inner_dict[inner_key] = inner_val
			attrs_dict[key] = inner_dict
		else:
			attrs_dict[key] = val
	business[ATTRIBUTES] = attrs_dict

def add_cities(cities, business):
	"""
	add business's city to cities set
	"""
	cities.add(business[CITY])

def add_categories(categories, business):
	"""
	add business's categories to categories set
	"""
	for category in business[CATEGORIES]:
		categories.add(category)

def add_attributes(attributes, business):
	"""
	add business's attributes to attributes dictionary
	"""
	attrs = business[ATTRIBUTES]
	for (key, val) in attrs.items():
		if isinstance(val, dict):
			inner_dict = attributes.get(key, dict())
			for (inner_key, inner_val) in val.items():
				inner_vals = inner_dict.get(inner_key, set())
				inner_vals.add(inner_val)
				inner_dict[inner_key] = inner_vals
			attributes[key] = inner_dict
		else:
			vals = attributes.get(key, set())
			vals.add(val)
			attributes[key] = vals

def init_24_hours(interval=5):
	"""
	interval : we only consider minute dividable by interval
			since no restaurant will open at 10:01

	return list of all possible time in 24 hour scale
	"""
	res = list()

	for hour in range(24):
		for minute in range(60):
			if minute % interval == 0:
				res.append("{}:{}".format(hour, minute))

	return res

def is_boolean_set(obj_set):
	"""
	determine whether or not the obj_set only contains boolean
	"""
	for obj in obj_set:
		if not isinstance(obj, bool):
			return False
	return True

def handle_attributes_features(
	tffeatures,
	valfeatures,
	attributes,
	default_value=DEFAULT_VAL_IF_NOT_EXIST,
	feature_separator=FEATURE_SEPARATOR
	):
	"""
	determine each attribute as tffeature or valfeature

	tffeature: list of tffeature
	valfeature: mapping from valfeature to values
	attributes: mapping from attributes to values
	default_value: if some attribute is valfeature, append values with default_value
	feature_separator: for complex feature name
	"""
	for (key, val) in attributes.items():
		if (isinstance(val, dict)):
			for (inner_key, inner_val) in val.items():
				feature_name = ATTRIBUTES + feature_separator + key + feature_separator + inner_key

				if is_boolean_set(inner_val):
					# inner feature is true/false feature
					tffeatures.append(feature_name)
				else:
					# inner feature is val feature
					inner_val.add(default_value)
					valfeatures[feature_name] = inner_val
		else:
			feature_name = ATTRIBUTES + feature_separator + key

			if is_boolean_set(val):
				# feature is true/false feature
				tffeatures.append(feature_name)
			else:
				# feature is val feature
				val.add(default_value)
				valfeatures[feature_name] = val

def init_features(
	attributes,
	categories,
	cities,
	workdays,
	start_time,
	end_time,
	default_value=DEFAULT_VAL_IF_NOT_EXIST,
	feature_separator=FEATURE_SEPARATOR
	):
	"""
	initialize feature from attributes, categories, cities, 
	workdays, start_time & end_time

	output:
		tffeature : list of true / false feature
		valfeature : mapping from value feature to values
		existfeature : list of exist feature
	"""
	tffeatures = list()
	valfeatures = dict()
	existfeatures = list()

	# categories are exist feature
	for category in categories:
		feature_name = CATEGORIES + feature_separator + category
		existfeatures.append(feature_name)

	# cities are val feature
	cities.add(DEFAULT_VAL_IF_NOT_EXIST)
	valfeatures[CITY] = cities

	# workdays are exist feature
	for day in workdays:
		feature_name = WORKDAYS + feature_separator + day
		existfeatures.append(feature_name)

	# day_start, day_end are val features
	time_list = init_24_hours()
	time_list.append(DEFAULT_VAL_IF_NOT_EXIST)
	for day_start in start_time:
		valfeatures[day_start] = time_list

	for day_end in end_time:
		valfeatures[day_end] = time_list

	# attributes can be val features or true false features
	handle_attributes_features(tffeatures, valfeatures, attributes)
	return tffeatures, valfeatures, existfeatures

def apply_queries(
	restaurant, 
	qs, 
	default_value_if_not_found,
	target_type):
	"""
	restaurant: json object restaurant
	qs: queries
	default_value_if_not_found: default values
	target_type: target type
	"""
	res = restaurant
	for i in range(len(qs)):
		if qs[i] not in res:
			return default_value_if_not_found
		res = res[qs[i]]

	if not isinstance(res, target_type):
		return default_value_if_not_found

	return res

def exist_queries(restaurant, qs):
	"""
	restaurant: json object restaurant
	qs: queries
	"""
	res = restaurant
	for i in range(len(qs) - 1):
		res = res[qs[i]]
	return qs[len(qs) - 1] in res

def map_restaurant_to_feature_dict(
	restaurant, 
	tffeature, 
	valfeature, 
	existfeature,
	feature_separator=FEATURE_SEPARATOR,
	default_value=DEFAULT_VAL_IF_NOT_EXIST,
	default_boolean=False
	):
	"""
	for input restaurant, output the feature vector given

		tffeature : list of true / false feature
		valfeature : mapping from value feature to values
		existfeature : list of exist feature

	return mapping from each feature to value
	"""
	feature_dict = dict()
	for key in tffeature:
		feature_dict[key] = apply_queries(
			restaurant, 
			key.split(feature_separator), 
			default_boolean, 
			bool
			)

	for key in valfeature:
		feature_dict[key] = apply_queries(
			restaurant, 
			key.split(feature_separator), 
			default_value, 
			str
			)

	for key in existfeature:
		feature_dict[key] = exist_queries(
			restaurant, 
			key.split(feature_separator)
			) is not None

	return frozenset(feature_dict.items())

#-------------------------------
# Main Functions
#-------------------------------
def load_restaurants_and_features(
	business_file,
	feature_separator=FEATURE_SEPARATOR
	):
	"""
	business_file: business_file: path to access business.json 

	return:
		restaurants: list of json objects for each restaurants
		tffeature: list of true/false features
		valfeature: dict of value features to values
		existfeature: list of exist featuers
	"""
	restaurants = list()
	attributes = dict()
	categories = set()
	cities = set()

	workdays = WORKDAYS_VALUE
	start_time = [WORKDAYS_START(day) for day in workdays]
	end_time = [WORDDAYS_END(day) for day in workdays]

	try:
		with open(business_file, "r") as file:
			lines = file.readlines()
			for line in lines:
				business = json.loads(line)
				if (is_valid_business(business) 
					and is_restaurant(business) 
					and is_open(business)):
					# load this restaurant
					preprocess_categories(business)
					preprocess_hours_extend_workdays(business)
					preprocess_attributes(business)
					
					restaurants.append(business)

					add_attributes(attributes, business)
					add_categories(categories, business)
					add_cities(cities, business)
	except IOError:
		print("{} doesn't exists".format(business_file))

	print("distinct restaurants {}".format(len(restaurants)))

	tffeature, valfeature, existfeature = init_features(
		attributes, 
		categories, 
		cities, 
		workdays,
		start_time,
		end_time)
	print("distinct true/false features {}".format(len(tffeature)))
	print("distinct value features {}".format(len(valfeature)))
	print("distinct exist features {}".format(len(existfeature)))
	return restaurants, tffeature, valfeature, existfeature

def develop_feature_ids(
	restaurants,
	tffeature, 
	valfeature,
	existfeature,
	feature_separator=FEATURE_SEPARATOR):
	"""
	restaurants: list of restaurants json object
	tffeature: list of true/false features
	valfeature: dict of value features to values
	existfeature: list of exist features

	return:
		feature_vector_to_int: mapping from feature_dict to feature_id
		restaurants_id_to_int: mapping from business_id to feature_id
	"""
	feature_vector_to_int = dict()
	restaurants_id_to_int = dict()
	count_feature_id = 0

	duplicate_restaurants = dict()
	max_dup = 0

	for restaurant in restaurants:
		feature_dict = map_restaurant_to_feature_dict(
			restaurant,
			tffeature,
			valfeature,
			existfeature,
			feature_separator=feature_separator
			)

		if feature_dict in feature_vector_to_int:
			restaurants_id_to_int[restaurant[RESTAURANT_ID]] = feature_vector_to_int[feature_dict]
			duplicate_restaurants[feature_vector_to_int[feature_dict]] = \
				duplicate_restaurants.get(feature_vector_to_int[feature_dict], 1) + 1
			max_dup = max(max_dup, duplicate_restaurants[feature_vector_to_int[feature_dict]])
			continue

		count_feature_id += 1
		feature_vector_to_int[feature_dict] = count_feature_id
		restaurants_id_to_int[restaurant[RESTAURANT_ID]] = count_feature_id

	print("Up-to-date feature_ids: {}".format(count_feature_id))
	print("Num of feature_id with multiple restaurants: {}".format(len(duplicate_restaurants)))
	print("Max duplicate restaurants to same feature_id: {}".format(max_dup))
	return feature_vector_to_int, restaurants_id_to_int

def load_reviews_and_users(
	review_file,
	restaurants_id_to_int
	):
	"""
	review_file: path to access review.json
	restaurants_id_to_int: mapping from business_id to feature_id

	return:
		user_id_to_int: mapping from user_id to int_user_id
		filtered_reviews: list of reviews [int_user_id, feature_id, rating, timestamp]
		original_reviews: list of reviews using original restaurant id [user_id, restaurant_id, rating, timestamp]
	"""
	user_id_to_int = dict()

	# remember all latest rating on user - restaurant pair
	user_restaurant_timestamp_rating = dict()

	count_user_id = 0

	try:
		with open(review_file, "r") as file:
			lines = file.readlines()
			for line in lines:
				line = json.loads(line)

				restaurant_id = line[RESTAURANT_ID]
				if restaurant_id not in restaurants_id_to_int:
					continue

				rating = line[RATING]
				timestamp = int(datetime.strptime(line[TIMESTAMP], '%Y-%m-%d %H:%M:%S').timestamp())

				# get the int type user_id
				user_id = line[USER_ID]
				if user_id in user_id_to_int:
					user_id = user_id_to_int[user_id]
				else:
					count_user_id += 1
					user_id_to_int[user_id] = count_user_id
					user_id = count_user_id

				if user_id not in user_restaurant_timestamp_rating:
					user_restaurant_timestamp_rating[user_id] = dict()

				if restaurant_id not in user_restaurant_timestamp_rating[user_id]:
					user_restaurant_timestamp_rating[user_id][restaurant_id] = [timestamp, rating]
				else:
					# update the rating to latest rating
					if timestamp > user_restaurant_timestamp_rating[user_id][restaurant_id][0]:
						user_restaurant_timestamp_rating[user_id][restaurant_id] = [timestamp, rating]

	except IOError:
		print("{} doesn't exists".format(review_file))

	filtered_reviews = list()
	original_reviews = list()
	for (user, rest_rating) in user_restaurant_timestamp_rating.items():
		for (restaurant, latest_rating) in rest_rating.items():
			timestamp = latest_rating[0]
			rating = latest_rating[1]

			filtered_reviews.append([user, restaurants_id_to_int[restaurant], rating, timestamp])
			original_reviews.append([user, restaurant, rating, timestamp])

	print("Up-to-date user counts: {}".format(count_user_id))
	print("Up-to-date reviews on restaurants: {}".format(len(filtered_reviews)))
	return user_id_to_int, filtered_reviews, original_reviews

def evaluate_feature_ids(
	filtered_reviews,
	original_reviews,
	restaurants_id_to_int,
	output_dir,
	threshold=2
	):
	"""
	filtered_reviews: list of reviews [int_user_id, feature_id, rating, timestamp]
	original_reviews: list of reviews using original restaurant id [user_id, restaurant_id, rating, timestamp]
	restaurants_id_to_int: mapping from business_id to feature_id
	output_dir: output directory
	threshold: max rating difference allowed for single user_feature pair

	output:
		files for each invalid user_feature pair (max rating diff > threshold)
		containing restaurants on which they disagree

	return:
		max difference of rating on single feature id and single user
	"""
	user_restaurant_max = dict()
	user_restaurant_min = dict()
	user_restaurant_bucket_recorder = dict()

	for i in range(len(filtered_reviews)):
		review = filtered_reviews[i]
		user = review[0]
		restaurant = review[1]
		rating = review[2]
		timestamp = review[3]

		if user not in user_restaurant_max:
			user_restaurant_max[user] = dict()
			user_restaurant_min[user] = dict()
			user_restaurant_bucket_recorder[user] = dict()

		if restaurant not in user_restaurant_max[user]:
			user_restaurant_max[user][restaurant] = rating
			user_restaurant_min[user][restaurant] = rating
			user_restaurant_bucket_recorder[user][restaurant] = dict()
		else:
			user_restaurant_max[user][restaurant] = max(rating, user_restaurant_max[user][restaurant])
			user_restaurant_min[user][restaurant] = min(rating, user_restaurant_min[user][restaurant])

		if rating not in user_restaurant_bucket_recorder[user][restaurant]:
			user_restaurant_bucket_recorder[user][restaurant][rating] = set()
		user_restaurant_bucket_recorder[user][restaurant][rating].add(original_reviews[i][1])

	max_diff = 0
	disagree_user = 0
	bad_feature_ids = set()

	filedir = "{}{}".format(output_dir, datetime.today().strftime('%Y-%m-%d-%H:%M:%S/'))
	os.makedirs(os.path.dirname(filedir), exist_ok=True)

	for (user, val) in user_restaurant_max.items():
		is_disagree_user = False
		for (restaurant, max_rating) in val.items():
			min_rating = user_restaurant_min[user][restaurant]
			max_diff = max(max_diff, max_rating - min_rating)

			if max_rating - min_rating > threshold:
				is_disagree_user = True
				bad_feature_ids.add(restaurant)

				filename = "{}{}_{}.txt".format(
					filedir,
					user, 
					restaurant
					)

				with open(filename, "w") as file:
					for (key, vals) in user_restaurant_bucket_recorder[user][restaurant].items():
						file.write("user rated restaurant with score = {}: {}\n".format(
							key, 
							len(user_restaurant_bucket_recorder[user][restaurant][key])
							))

					for (key, vals) in user_restaurant_bucket_recorder[user][restaurant].items():
						file.write("score = {} restaurants:\n {}\n".format(
							key, 
							user_restaurant_bucket_recorder[user][restaurant][key]
							))

		if is_disagree_user:
			disagree_user += 1

	print("Max difference between single user and single feature id: {}".format(max_diff))
	print("Given threshold {}, number of users disagree with oneself: {}".format(
		threshold, 
		disagree_user
		))
	print("Given threshold {}, number of bad feature ids are: {}\n{}\n".format(
		threshold,
		len(bad_feature_ids),
		bad_feature_ids
		))
	return bad_feature_ids

def load_dedup_filtered_reviews(
	filtered_reviews,
	dedup_by="latest"
	):
	"""
	filtered_reviews: list of reviews [int_user_id, feature_id, rating, timestamp]
	dedup_by: can be "latest" or "average"

	dedup by "latest":
		for each duplicated user_feature pair
			we only take the latest rating

	dedup by "average":
		for each duplicated user_feature pair:
			we will take the latest timestamp & average rating		

	return:
		dedup_filtered_reviews: list of reviews [int_user_id, feature_id, rating, timestamp]
	"""
	if dedup_by != "latest" and dedup_by != "average":
		raise ValueError("no such deduplication method: {}".format(dedup_by))

	user_restaurant_timestamp_rating = dict()
	for review in filtered_reviews:
		user_id = review[0]
		restaurant_id = review[1]
		rating = review[2]
		timestamp = review[3]

		if user_id not in user_restaurant_timestamp_rating:
			user_restaurant_timestamp_rating[user_id] = dict()

		if restaurant_id not in user_restaurant_timestamp_rating[user_id]:
			user_restaurant_timestamp_rating[user_id][restaurant_id] = [[timestamp, rating]]
		else:
			# update the rating to latest rating
			if timestamp > user_restaurant_timestamp_rating[user_id][restaurant_id][0][0]:
				user_restaurant_timestamp_rating[user_id][restaurant_id].insert(0, [timestamp, rating])
			else:
				user_restaurant_timestamp_rating[user_id][restaurant_id].append([timestamp, rating])

	dedup_by_latest = dedup_by == "latest"
	dedup_filtered_reviews = list()
	for (user, rest_rating) in user_restaurant_timestamp_rating.items():
		for (restaurant, latest_rating) in rest_rating.items():
			if dedup_by_latest:	
				timestamp = latest_rating[0][0]
				rating = latest_rating[0][1]
			else:
				sum_rating = 0
				count_rating = 0
				for [timestamp, rating] in latest_rating:
					sum_rating += rating
					count_rating += 1
				
				timestamp = latest_rating[0][0]
				rating = sum_rating / count_rating

			dedup_filtered_reviews.append([user, restaurant, rating, timestamp])

	return dedup_filtered_reviews


def map_restaurants_to_cities(
	restaurants,
	restaurants_id_to_int
	):
	"""
	restaurants: list of json objects for each restaurants
	restaurants_id_to_int: mapping from business_id to feature_id

	return:
		feature_id_to_cities: mapping from feature_id to city
	"""
	feature_id_to_cities = dict()

	for restaurant in restaurants:
		feature_id = restaurants_id_to_int[restaurant[RESTAURANT_ID]]
		city = restaurant[CITY]

		feature_id_to_cities[feature_id] = city

	return feature_id_to_cities

def map_users_to_cities(
	reviews,
	feature_id_to_cities
	):
	"""
	reviews: list of reviews [int_user_id, feature_id, rating, timestamp]
	feature_id_to_cities: mapping from feature_id to city

	return:
		int_user_id_to_cities: mapping from int_user_id to set of cities user has rated
	"""
	int_user_id_to_cities = dict()

	for review in reviews:
		user = review[0]
		restaurant = review[1]
		rating = review[2]
		timestamp = review[3]

		if user not in int_user_id_to_cities:
			int_user_id_to_cities[user] = set()

		int_user_id_to_cities[user].add(feature_id_to_cities[restaurant])

	return int_user_id_to_cities


def save_filtered_open_restaurants(
	restaurants,
	output_dir,
	output_file="filtered_open_retaurants.json"
	):
	"""
	restaurants: list of json objects for each restaurants
	output_dir: output directory
	output_file: output file name

	output:
	e.g.
	{"business_id": "pQeaRpvuhoEqudo3uymHIQ", ...., "workdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
	"""
	filename = output_dir + output_file
	with open(filename, "w") as file:
		for restaurant in restaurants:
			file.write("{}\n".format(json.dumps(restaurant)))


def save_features(
	features,
	output_dir,
	output_file,
	save_mapping=False,
	key_val_separator=KEY_VAL_SEPARATOR
	):
	"""
	features: features
	output_dir: output directory
	output_file: output file name
	save_mapping: whether or not features is a mapping
	key_val_separator: separate between feature and value if save_mapping

	output:
		each line is item in feature list
	"""
	filename = output_dir + output_file
	if save_mapping:
		with open(filename, "w") as file:
			for (feature_name, vals) in features.items():
				for val in vals:
					file.write("{}{}{}\n".format(feature_name, key_val_separator, val))
	else:
		with open(filename, "w") as file:
			for feature_name in features:
				file.write("{}\n".format(feature_name))


def save_feature_vec_to_int(
	feature_vector_to_int,
	output_dir,
	output_file="feature_vector_to_int.json"
	):
	"""
	feature_vector_to_int: mapping from feature_dict to feature_id
	output_dir: output directory
	output_file: output file name

	output:
		each line is json object contains feature_id and feature_dict
	"""
	filename = output_dir + output_file
	with open(filename, "w") as file:
		for (feature_dict, feature_id) in feature_vector_to_int.items():
			item = dict()
			item[FEATURE_ID] = feature_id
			item[FEATURE_DICT] = dict(feature_dict)
			file.write("{}\n".format(json.dumps(item)))


def save_mapping(
	mapping,
	value_is_set,
	output_dir,
	output_file,
	col_separator=COL_SEPARATOR
	):
	"""
	mapping: one-on-one mapping in dictionary
	value_is_set: whether or not value is set
	output_dir: output directory
	output_file: output file name
	col_separator: separator between key and values

	output:
		each line is (key, value) pair separated by col_separator
		if value_is_set, then create pairs for each val in value set
	"""
	filename = output_dir + output_file
	if value_is_set:
		with open(filename, "w") as file:
			for (key, vals) in mapping.items():
				for val in vals:
					file.write("{}{}{}\n".format(key, col_separator, val))
	else:
		with open(filename, "w") as file:
			for (key, val) in mapping.items():
				file.write("{}{}{}\n".format(key, col_separator, val))


def save_reviews(
	reviews,
	output_dir,
	output_file,
	col_separator=COL_SEPARATOR
	):
	"""
	reviews: list of reviews [int_user_id, feature_id, rating, timestamp]
	output_dir: output directory
	output_file: output file name
	col_separator: separator between key and values

	output:
		each line is col_separator separated review [int_user_id, feature_id, rating, timestamp]
	"""
	filename = output_dir + output_file
	with open(filename, "w") as file:
		for review in reviews:
			file.write("{}{}{}{}{}{}{}\n".format(
				review[0], col_separator,
				review[1], col_separator,
				review[2], col_separator,
				review[3]))

def preprocess_yelp_open_dataset(
	business_file,
	review_file,
	output_dir,
	feature_separator=FEATURE_SEPARATOR,
	col_separator=COL_SEPARATOR
	):
	"""
	business_file: path to access business.json 
	review_file: path to access review.json
	output_dir: path to save output files

	output files:

	filtered_open_retaurants.json:
	only contains business that are open restaurants
	appended the orignal business json with needed features
	e.g.
	{"business_id": "pQeaRpvuhoEqudo3uymHIQ", ...., "workdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}

	tffeature.data:
	true/false feature to define a restaurants
	each line is a true/false feature
	e.g.
	attributes::BusinessParking::valet

	valfeature.data:
	value feature to define a restaurants
	for each line of feature, last column is value
	e.g.
	city::St. Léonard

	existfeature.data:
	exist feature to define a restaurants
	for each line of feature, last column is value appear in a list
	e.g.
	categories::Jazz & Blues

	user_id_to_int.data:
	map user_id to int_user_id
	e.g.
	RR-2nouBn408e3djxC470g	9

	restaurants_id_to_int.data:
	map business_id to int feature_id
	e.g.
	vjTVxnsQEZ34XjYNS-XUpA	4

	feature_vector_to_int.json:
	map restaurant feature vector to feature_id
	each line is json object contains feature_id and feature_dict

	filtered_reviews.data:
	each line is a rating behavior:
	int_user_id, feature_id, rating, timestamp
	e.g.
	1	93	5.0	1575497346

	int_user_id_to_cities.data:
	this file is for inferring users' location from reviews
	e.g.
	9	St. Léonard

	feature_id_to_cities.data:
	e.g.
	4	St. Léonard
	"""
	# load from original dataset
	os.makedirs(os.path.dirname(output_dir), exist_ok=True)
	eval_feature_dir = output_dir + "eval_feature_ids/"
	os.makedirs(os.path.dirname(eval_feature_dir), exist_ok=True)

	restaurants, tffeature, valfeature, existfeature = load_restaurants_and_features(BUSINESS)

	feature_vector_to_int, restaurants_id_to_int = develop_feature_ids(
		restaurants=restaurants, 
		tffeature=tffeature, 
		valfeature=valfeature, 
		existfeature=existfeature
		)

	user_id_to_int, filtered_reviews, original_reviews = load_reviews_and_users(
		review_file=REVIEW, 
		restaurants_id_to_int=restaurants_id_to_int
		)

	# study features
	bad_feature_ids = evaluate_feature_ids(
		filtered_reviews=filtered_reviews, 
		original_reviews=original_reviews, 
		restaurants_id_to_int=restaurants_id_to_int, 
		output_dir=eval_feature_dir,
		threshold=2
		)

	dedup_filtered_reviews = load_dedup_filtered_reviews(
		filtered_reviews=filtered_reviews, 
		dedup_by="average"
		)

	# load city info
	feature_id_to_cities = map_restaurants_to_cities(
		restaurants=restaurants, 
		restaurants_id_to_int=restaurants_id_to_int
		)

	int_user_id_to_cities = map_users_to_cities(
		reviews=dedup_filtered_reviews, 
		feature_id_to_cities=feature_id_to_cities
		)

	save_filtered_open_restaurants(
		restaurants=restaurants,
		output_dir=output_dir,
		output_file="filtered_open_retaurants.json"
		)

	save_features(
		features=tffeature,
		output_dir=output_dir,
		output_file="tffeature.data",
		save_mapping=False
		)

	save_features(
		features=existfeature,
		output_dir=output_dir,
		output_file="existfeature.data",
		save_mapping=False
		)

	save_features(
		features=valfeature,
		output_dir=output_dir,
		output_file="valfeature.data",
		save_mapping=True
		)

	save_feature_vec_to_int(
		feature_vector_to_int=feature_vector_to_int,
		output_dir=output_dir,
		output_file="feature_vector_to_int.json"
		)

	save_mapping(
		mapping=restaurants_id_to_int, 
		value_is_set=False, 
		output_dir=output_dir, 
		output_file="restaurants_id_to_int.data"
		)

	save_mapping(
		mapping=user_id_to_int, 
		value_is_set=False, 
		output_dir=output_dir, 
		output_file="user_id_to_int.data"
		)

	save_reviews(
		reviews=filtered_reviews,
		output_dir=output_dir,
		output_file="filtered_reviews.data"
		)

	save_reviews(
		reviews=dedup_filtered_reviews,
		output_dir=output_dir,
		output_file="dedup_filtered_reviews.data"
		)

	save_reviews(
		reviews=original_reviews,
		output_dir=output_dir,
		output_file="original_reviews.data"
		)
	
	save_mapping(
		mapping=feature_id_to_cities, 
		value_is_set=False, 
		output_dir=output_dir, 
		output_file="feature_id_to_cities.data"
		)

	save_mapping(
		mapping=int_user_id_to_cities, 
		value_is_set=True, 
		output_dir=output_dir, 
		output_file="int_user_id_to_cities.data"
		)
