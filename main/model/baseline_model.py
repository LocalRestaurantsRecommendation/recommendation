import pandas as pd
from main.constants import *
from main.model.model_utils import init_user_item

def rank_restaurants_by_rates(data):
	"""
	rank restaurants by num of ratings received
	"""
	item_counts = data[COL_ITEM].value_counts().to_frame().reset_index()
	item_counts.columns = [COL_ITEM, COL_PREDICTION]
	return item_counts

def get_prediction(item_counts, user_item):
	"""
	Get prediction scores on user_item pairs

		item_counts: pandas dataframe, 
			item_score mapping for baseline model
		user_item: pandas dataframe,
			user_item pairs to score
	"""
	baseline_recommendations = pd.merge(item_counts, user_item, on=[COL_ITEM], how='inner')
	return baseline_recommendations

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
		self.user = user_id
		self.name = model_name
		self.data = data_reviews

	def close(self):
		"""
		do something when close this model
		"""
		pass

	def train(self):
		"""
		train on hyperparmaters
		"""
		self.item_counts = rank_restaurants_by_rates(self.data)


	def predict(self, k=10, removeSeen=True):
		"""
		predict top 10 restaurants on user

		if k <= 0, return all predictions
		"""
		user_item = init_user_item(user_id=self.user, data=self.data, removeSeen=removeSeen)
		baseline_recommendations = get_prediction(self.item_counts, user_item)
		res = baseline_recommendations[baseline_recommendations[COL_USER] == self.user] \
				.sort_values(COL_PREDICTION, ascending=False)

		if k <= 0:
			return res
		else:
			return res.head(k)

	def save_model(self, output_dir):
		"""
		save this model under output_dir with self.<model_name>_<current_time>
		"""
		pass