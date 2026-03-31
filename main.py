import warnings
warnings.simplefilter('ignore', category=FutureWarning)
import pandas as pd
from datetime import datetime
from pathlib import Path

# FindSpark simplifies the process of using Apache Spark with Python
import findspark
findspark.init()


from pyspark.ml.regression import LinearRegression, FMRegressor, RandomForestRegressor, GBTRegressor
from spark_utils import  train, load_model, multimetric_evaluator, build_spark


def run(models_list=None, train_data=None,
         val_data=None, test_data=None, path_to_save=None, metrics=None):


    train_res = {}
    test_res = {}
    for m in models_list:
        res, weights_path, model_name = train(train_data, val_data, model=m,
                                                  path=path_to_save, metrics_list=metrics)
        print('.......Testing.......')
        model = load_model(cross_validated=False, path=weights_path)
        test_preds = model.transform(test_data)
        t_res = multimetric_evaluator(test_preds, metrics_list)
        test_res[m] = t_res
        train_res[m] = res

    train_performance_df = pd.DataFrame(train_res).reset_index()
    test_performance_df = pd.DataFrame(test_res).reset_index()

    return train_performance_df, test_performance_df


if __name__ == '__main__':


    spark = build_spark()

    base_path = Path(__file__).resolve().parent

    data_path = base_path /'dataset' /'NASA_airfoil_noise_cleaned.parquet'

    df = spark.read.parquet(f'{data_path}')

    lr = LinearRegression(featuresCol="scaledFeatures",
                          labelCol="SoundLevelDecibels",
                          regParam=0.2,
                          elasticNetParam=0.8)
    fmr =FMRegressor(featuresCol="scaledFeatures",
                     labelCol="SoundLevelDecibels",
                     stepSize=0.25)
    rfr = RandomForestRegressor(featuresCol="scaledFeatures",
                                labelCol="SoundLevelDecibels",
                                maxDepth= 8)
    gbr = GBTRegressor(featuresCol="scaledFeatures",
                       labelCol="SoundLevelDecibels",
                       maxDepth=8)

    # Split data
    (trainingData, testingData) = df.randomSplit([0.7, 0.3], seed=42)
    (testingData, valData) = testingData.randomSplit([0.6, 0.4], seed=42)

    # Create outtputs directories
    outputs = ['models', 'metrics']

    output_dir = {}
    for o in outputs:
        output_dir[o] = base_path / 'results' / o
        output_dir[o].mkdir(exist_ok=True)

    metrics_list = ['mae', 'mse', 'rmse', 'r2']

    train_metrics, test_metrics = run(models_list=[lr, fmr, rfr, gbr],
                                      train_data=trainingData,
                                      val_data=valData,
                                      test_data=testingData,
                                      path_to_save=output_dir['models'],
                                      metrics=metrics_list)

    spark_metrics = spark.createDataFrame(train_metrics)
    spark_metrics.show()
    spark_metrics = spark_metrics.withColumnRenamed("index", "Metric")
    spark_test_metrics = spark.createDataFrame(test_metrics)
    park_test_metrics = spark_test_metrics.withColumnRenamed("index", "Metric")

    # Save metrics results
    ts = datetime.timestamp(datetime.now())
    spark_metrics.write.mode("overwrite").parquet(f"{output_dir['metrics']}/training_metrics_{ts}.parquet")
    spark_test_metrics.write.mode("overwrite").parquet(f"{output_dir['metrics']}/test_metrics_{ts}.parquet")

    print("Training scores")
    spark_metrics.show()
    print("Test scores")
    spark_test_metrics.show()

    # Stop context and session
    spark.stop()