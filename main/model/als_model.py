import os
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, LongType
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import desc

from main.constants import *
from main.model.model_utils import init_user_item

def envr_check():
	os.environ["PYSPARK_PYTHON"]="python3"
	os.environ["PYSPARK_DRIVER_PYTHON"]="python3"
	os.environ["SPARK_LOCAL_IP"]="localhost"

def start_or_get_spark(app_name="Sample", url="local[*]", memory="10g"):
	spark = SparkSession.builder.master(url).appName(app_name).getOrCreate()
	sc = spark.sparkContext
	checkpoint_dir = PROJECT_PATH + 'checkpoint/'
	sc.setCheckpointDir(checkpoint_dir)
	return spark, sc

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
		
		envr_check()

		self.spark = None
		self.sc = None
		self.spark, self.sc = start_or_get_spark(app_name="ALS Model", memory="16g")
		self.schema = StructType(
							(
								StructField(COL_USER, IntegerType()),
								StructField(COL_ITEM, IntegerType()),
								StructField(COL_RATING, FloatType()),
								StructField(COL_TIMESTAMP, LongType()),
							)
						)

		self.data_train = self.spark.createDataFrame(
			self.data,
			schema=self.schema
			)

		self.maxIter = 10
		self.rank = 10
		self.regParam = 0.001
		self.coldStartStrategy = "drop"
		self.nonnegative = True
		self.seed = 42

	def close(self):
		"""
		do something when close this model
		"""
		if self.spark is not None:
			self.spark.stop()
			self.sc.stop()
		
	def train(self):
		"""
		train on hyperparmaters
		"""
		self.als = ALS(
			maxIter=self.maxIter, 
			rank=self.rank,
			regParam=self.regParam,
			userCol=COL_USER,
			itemCol=COL_ITEM, 
			ratingCol=COL_RATING, 
			coldStartStrategy=self.coldStartStrategy,
			nonnegative=self.nonnegative,
			seed=self.seed
			)

		self.model = self.als.fit(self.data_train)

	def predict(self, k=10, removeSeen=True):
		"""
		predict top 10 restaurants on user

		if k <= 0, return all predictions
		"""
		user_item = init_user_item(user_id=self.user, data=self.data, removeSeen=removeSeen)

		schema_user_item = StructType(
								(
									StructField(COL_USER, IntegerType()),
									StructField(COL_ITEM, IntegerType())
								)
							)
		user_item = self.spark.createDataFrame(user_item, schema=schema_user_item)
		sdf_pred = self.model.transform(user_item)
		pdf_pred = sdf_pred.where(f"{COL_USER} == {self.user}") \
					.orderBy(desc(COL_PREDICTION))

		if k <= 0:
			# be careful that it's cost to convert huge table
			return pdf_pred.toPandas()
		else:
			return pdf_pred.limit(k).toPandas()

	def save_model(self, output_dir):
		"""
		save this model under output_dir with self.<model_name>_<current_time>
		"""
		pass