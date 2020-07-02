import pandas as pd
from main.constants import *
from main.model.als_model import Model as AlsModel
from main.model.baseline_model import rank_restaurants_by_rates

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

		item_counts = rank_restaurants_by_rates(self.data)
		df = pd.merge(item_counts, self.data, on=[COL_ITEM], how='inner')
		df["result"] = df[COL_RATING] * df[COL_PREDICTION]
		df = df[[COL_USER, COL_ITEM, "result", COL_TIMESTAMP]].rename(columns={"result" : COL_RATING})

		self.model = AlsModel(
			user_id=self.user,
			model_name=f"{self.name}_als",
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
		return self.model.predict(k=k, removeSeen=removeSeen)

	def save_model(self, output_dir):
		"""
		save this model under output_dir with self.<model_name>_<current_time>
		"""
		pass