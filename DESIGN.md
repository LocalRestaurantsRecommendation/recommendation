# Goals
* turn the files dataset into friendly version to help python training
* train the model from following dataset
* Provide top 10 restaurants users might never know but will probably like.

# Relationship With Other Systems

```
			Full Stack Recommendation
				|
				|
				|
				|
	-------------------------Back End--------------------------------
	|				|				|
	|				|				|
	|				|				|
	MySQL:database		Python: Recommendation		Java: Servlets
						<------------------------->
							preprocess(files)
							train(files)
							predict(user)
	       <------------------------------------------------->
		R/W into database
		Execute to local files
```

# Structure of Python Project

```
main/
   |
   |__ constants.py: contains contants values
   |
   |__ preprocessor.py: place to hold preprocess function
   |
   |__ data_loader.py: helper functions for loading data from files
   |
   |__ recommendar.py: starting point of recommendation
   |
   |__ evaluation.py: helper functions for evaluations
   |
   |__ evaluate_models.py: evaluate different models on yelp open dataset
   |
   |__ data/: place to hold preprocessed data
   |
   |__ output/: place to hold training results
   |
   |__ model/: place to hold different models
		|
		|__ baseline_model.py
		|
		|__ als_model.py
		|
		|__ naive_hybrid_baseline_als_model.py
		|
		|__ time_biased_hybrid_model.py
		|
		|__ model_utils.py : place to hold helper functions for models
   
```

## preprocessor.py
```python
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

	tf_feature.data:
	true/false feature to define a restaurants
	each line is a true/false feature
	e.g.
	attributes::BusinessParking::valet

	val_feature.data:
	value feature to define a restaurants
	for each line of feature, last column is value
	e.g.
	city::St. Léonard

	exist_feature.data:
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
def develop_feature_ids(
	restaurants,
	tffeature, 
	valfeature,
	existfeature,
	feature_separator=FEATURE_SEPARATOR
	):
	"""
	restaurants: list of restaurants json object
	tffeature: list of true/false features
	valfeature: dict of value features to values
	existfeature: list of exist features

	return:
		feature_vector_to_int: mapping from feature_dict to feature_id
		restaurants_id_to_int: mapping from business_id to feature_id
	"""

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
	"""

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

def save_reviews(
	reviews,
	output_dir,
	output_file,
	col_separator=COL_SEPARATOR
	):
	"""
	reviews: list of reviews [int_user_id, restaurant, rating, timestamp]
	output_dir: output directory
	output_file: output file name
	col_separator: separator between key and values

	output:
		each line is col_separator separated review [int_user_id, restaurant, rating, timestamp]
	"""
```

## data_loader.py
```python
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

def load_feature_vec_to_int(input_file):
	"""
	input_file: input_file_path
		each line is json object contains feature_id and feature_dict

	output:
		mapping from feature_dict to feature_id
	"""

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
```

## recommendar.py

```python
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
			int_user_id, best_train_number, average precision @ k, precision @ k, recall @ k
			1	10	0.9	0.5	0.5

		return average precision @ k for all users
		"""

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

	def predict(self, user_id):
		"""
		use all user's rating behavior up to now to predict future

		user_id: integer id for a user

		return:
			list of top k feature_ids
		"""
```

## evaluation.py
```python
def evaluate_top_k_for_user(
	df_truth,
	df_pred,
	k=TOP_K,
	threshold=THRESHOLD_SCALE_5,
	userCol=COL_USER,
	itemCol=COL_ITEM, 
	rateCol=COL_RATING,
	predCol=COL_PREDICTION
	):
	"""
	Evaluate precision at k, recall at k, average precision at k

		df_truth: pandas dataframe of groudtruth
		df_pred: pandas dataframe of prediction result
		k: precision at k
		threshold: consider item relevant if score > threshold
		userCol: column name of users
		itemCol: column name of items
		rateCol: column name of ratings score
		predCol: column name of prediction score

	return precision at k, recall at k, average precision as k
	"""
```

## evaluate_models.py
```python
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
```

## each model under model/ should contains
```python
class Model:
	"""
	state:
		self.user
		self.name
		self.data
		self.<hyperparameter_name> ...
	"""

	def __init__(self, user_id, model_name, data_reviews):
		"""
		user_id: 
			this model is for a specific user

		model_name: 
			give this model a name

		data_reviews:
			pandas dataframe where each row is review
		"""

	def close(self):
		"""
		do something when close this model
		"""

	def train(self):
		"""
		train on hyperparmaters
		"""

	def predict(self, k=10, removeSeen=True):
		"""
		predict top 10 restaurants on user

		if k <= 0, return all predictions
		"""

	def save_model(self, output_dir):
		"""
		save this model under output_dir with self.<model_name>_<current_time>
		"""
```