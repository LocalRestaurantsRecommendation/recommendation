import pandas as pd
from datetime import datetime
from main.constants import *
from main.model.naive_hybrid_baseline_als_model import Model as NHBAModel

RATING_TIME_RANGE = {
	"1 month": 31, 
	"1 week": 7,
	"3 days": 3,
	"1 day": 1
}

class Model:
	"""
	state:
		self.name
		self.data
		self.<hyperparameter_name> ...
	"""

	def __init__(self, model_name, data_reviews):
		"""
		model_name: give this model a name
		data_reviews:
			pandas dataframe where each row is review
		"""
		self.name = model_name
		self.data = data_reviews

		self.latest_rating_range = RATING_TIME_RANGE["1 week"]

		df = data_reviews.sort_values(COL_TIMESTAMP, ascending=False).reset_index()
		latest_timestamp = int(df.iloc[0][COL_TIMESTAMP])
		latest_datetime = datetime.fromtimestamp(latest_timestamp)

		first_invalid = 0
		for index, row in df.iterrows():
			cur_timestamp = int(row[COL_TIMESTAMP])
			cur_datetime = datetime.fromtimestamp(cur_timestamp)
			timediff = (latest_datetime - cur_datetime).days
			if timediff > self.latest_rating_range:
				first_invalid = index
				break

		df = df.head(first_invalid).reset_index()

		self.model = NHBAModel(
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

	def predict(self, user_id, k=10, removeSeen=True):
		"""
		predict top 10 restaurants on user

		if k <= 0, return all predictions
		"""
		if not removeSeen:
			return self.model.predict(user_id=user_id, k=k, removeSeen=removeSeen)

		naive_pred = self.model.predict(user_id=user_id, k=0, removeSeen=removeSeen)

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




