# Suppress warnings
from random import random


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
import pandas as pd

# FindSpark simplifies the process of using Apache Spark with Python
import findspark
findspark.init()

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.regression import LinearRegression, FMRegressor, RandomForestRegressor, GBTRegressor
from spark_utils import *


def run(label=None, pred_col=None, metrics_list=None, models_list=None, stages=None, train_data=None,
         val_data=None, test_data=None, path_to_save=None):

    evaluators_dict = evaluators(label=label, pred_col=pred_col, metrics_list=metrics_list)

    performance = {}
    test_performance = {}
    for model in models_list:
        metrics, weights_path, model_name = train(train_data, val_data, stages, model=model,
                                                  evaluators=evaluators_dict, path=path_to_save)
        performance.update(metrics)
        stages.pop()
        print('.......Testing.......')
        best_model = load_model(cross_validated=False, path=weights_path)
        test_prediction = best_model.transform(test_data)
        test_results = evaluation(evaluators_dict, test_prediction)
        test_performance[model_name] = test_results

    performance_df = pd.DataFrame(performance).reset_index(drop=False)
    test_performance_df = pd.DataFrame(test_performance).reset_index(drop=False)

    return performance_df, test_performance_df


if __name__ == '__main__':

     # Create Spark Context and SparkSession

     conf = SparkConf()
     conf.set("spark.executor.memory", "2g")
     sc = SparkContext(conf=conf)
     spark = SparkSession.builder.appName('Final Project').getOrCreate()
     sc.setLogLevel("error")

     data_path = 'bigdata/NASA_airfoil_noise_cleaned.parquet 12-17-47-019.parquet'
     df = spark.read.parquet(data_path)
     assembler = vect_assembler(df.columns[0:-1],'features')
     scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
     lr = LinearRegression(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels", regParam=0.2,
                           elasticNetParam=0.8)
     factRegressor = FMRegressor(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels", stepSize=0.3)
     randomForest = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels", maxDepth=8)
     gradientBoost = GBTRegressor(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels", maxDepth=8)

     # Split data
     (trainingData, testingData) = df.randomSplit([0.7, 0.3], seed=42)
     (testingData, valData) = testingData.randomSplit([0.6, 0.4], seed=42)

     objectives = ['mae', 'mse', 'rmse', 'r2']

     output_dir = "bigdata"
     models = [lr, factRegressor, randomForest, gradientBoost]

     train_metrics, test_metrics = run(label='SoundLevelDecibels', pred_col='prediction', metrics_list=objectives,
          models_list=models, stages=[assembler, scaler], train_data=trainingData, val_data=valData,
                                        test_data=testingData, path_to_save=output_dir)


     spark_metrics = spark.createDataFrame(train_metrics)
     spark_metrics = spark_metrics.withColumnRenamed("index", "Metric")
     spark_test_metrics = spark.createDataFrame(test_metrics)
     spark_test_metrics = spark_test_metrics.withColumnRenamed("index", "Metric")

     # Save metrics results
     ts = datetime.timestamp(datetime.now())
     spark_metrics.write.mode("overwrite").parquet(f"{output_dir}/training_metrics_{ts}.parquet")
     spark_test_metrics.write.mode("overwrite").parquet(f"{output_dir}/test_metrics_{ts}.parquet")

     print("Training scores")
     spark_metrics.show()
     print("Test scores")
     spark_test_metrics.show()

     # Stop context and session
     sc.stop()
     spark.stop()







