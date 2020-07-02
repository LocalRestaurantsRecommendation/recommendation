import pandas as pd
from datetime import datetime, timedelta
from main.constants import *
from main.model.naive_hybrid_baseline_als_model import Model as NHBAModel

DEFAULT_TIME_RANGE = 7
# tuned from [1,3,7,30] on 11 users (min_rating >= 500)

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

		self.latest_rating_range = timedelta(days = DEFAULT_TIME_RANGE)

		last_valid_time = data_reviews[data_reviews[COL_USER] == user_id] \
					.sort_values(COL_TIMESTAMP, ascending=False) \
					.iloc[0][COL_TIMESTAMP] - self.latest_rating_range.total_seconds()

		df = data_reviews[data_reviews[COL_TIMESTAMP] >= last_valid_time]

		self.model = NHBAModel(
			user_id=self.user,
			model_name=f"{self.name}_nhba",
			data_reviews=df
			)

	def close(self):
		"""
		do something when close this model
		"""
		self.model.close()

	def train(self):
		"""
		train on hyperparmaters
		"""
		self.model.train()

	def predict(self, k=10, removeSeen=True):
		"""
		predict top 10 restaurants on user

		if k <= 0, return all predictions
		"""
		if not removeSeen:
			return self.model.predict(k=k, removeSeen=removeSeen)

		naive_pred = self.model.predict(k=0, removeSeen=removeSeen)

		user_item_columns = [COL_USER, COL_ITEM]
		pred = naive_pred.loc[
					~naive_pred.set_index(user_item_columns).index.isin( 
						self.data.set_index(user_item_columns).index
					)
				]

		if k <= 0:
			return pred
		else:
			return pred.head(k)

	def save_model(self, output_dir):
		"""
		save this model under output_dir with self.<model_name>_<current_time>
		"""
		pass