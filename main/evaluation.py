import pandas as pd

from main.constants import *

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

	df_truth_thres = df_truth.loc[df_truth[COL_RATING] >= threshold]
	total_relevant = df_truth_thres.shape[0]

	hit_count = 0
	sum_precision = 0
	for index, row in df_pred.iterrows():
		if (index == k):
			break

		if ((df_truth_thres[COL_USER] == row[COL_USER]) 
			& (df_truth_thres[COL_ITEM] == row[COL_ITEM])).any():
			hit_count += 1
			sum_precision += hit_count / (index + 1)

	pk = hit_count / k

	if total_relevant == 0:
		rk = 0
	else:
		rk = hit_count / total_relevant

	if hit_count == 0:
		apk = 0
	else:
		apk = sum_precision / hit_count

	return pk, rk, apk