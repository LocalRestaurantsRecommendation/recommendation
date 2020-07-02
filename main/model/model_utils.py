import pandas as pd
import itertools
from main.constants import *

def init_user_item(
	user_id,
	data,
	removeSeen=True
	):
	"""
	generate user_item candidates given user_id and data

	user_id: user id for this user
	data: pandas dataframe of relevant reviews
	removeSeen: whether or not to remove seen data

	return pandas dataframe of user_item candidate pairs
	"""
	users = [user_id]
	items = data[COL_ITEM].unique()
	user_item_columns = [COL_USER, COL_ITEM]
	user_item_all = pd.DataFrame(
		list(itertools.product(users, items)),
		columns=user_item_columns
		)

	if removeSeen:
		user_item = user_item_all.loc[ 
						~user_item_all.set_index(user_item_columns).index.isin( 
							data.set_index(user_item_columns).index 
						)
					]
	else:
		user_item = user_item_all

	return user_item