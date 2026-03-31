from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from datetime import datetime
from pathlib import Path

from spark_utils import (build_spark, built_pipeline,
                         regression_evaluator,
                         eval_preds, load_model)


def regression_cv_optim(model, pipeline, n_folds=2, metric='rmse', data=None):

    # Create ParamGrid for Cross Validation
    grid = ParamGridBuilder().addGrid(model.maxDepth, [3, 5, 8, 10])\
            .addGrid(model.maxBins,[20, 25, 32])\
            .addGrid(model.maxIter, [20, 25, 30])\
            .build()

    # Get the model evaluator
    model_evaluator = regression_evaluator(metric)

    # Create n-fold CrossValidator
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=grid,
                        evaluator=model_evaluator,
                        numFolds=n_folds)

    # Run cross-validation
    return cv.fit(data)


if __name__ == "__main__":

    spark = build_spark()
    # Load the data
    try:
        base_dir = Path(__file__).resolve().parent
        data_path = base_dir / "dataset" / "NASA_airfoil_noise_cleaned.parquet"
        metrics_dir = base_dir / "results" / "metrics"
        models_dir = base_dir / "results" / "models"

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        df = spark.read.parquet(f"{data_path}")

        # Split data
        trainingData, testingData = df.randomSplit([0.7, 0.3], seed=42)

        gb_regressor = GBTRegressor(featuresCol="scaledFeatures",
                                    labelCol="SoundLevelDecibels")

        cv_pipeline = built_pipeline(trainingData, gb_regressor)

        cv_opt = regression_cv_optim(gb_regressor,
                                     cv_pipeline,
                                     n_folds=5,
                                     metric='rmse',
                                     data=trainingData)

        print(cv_opt)

        # Persist model
        cv_now = datetime.now()

        # convert from datetime to timestamp
        cv_ts = datetime.timestamp(cv_now)
        path = f'{models_dir}/optim_gboost_model_{cv_ts}'
        cv_opt.write().save(path)

        # Print the best model
        best_model = cv_opt.bestModel

        print('============Best Model Parameters============')

        print(f"MaxIter: {best_model.stages[-1].getMaxIter()}")
        print(f"MaxDepth: {best_model.stages[-1].getMaxDepth()}")
        print(f"MaxBins: {best_model.stages[-1].getMaxBins()}")
        print(f'Best RMSE {min(cv_opt.avgMetrics):.2f}')

        # Predict in testset
        cv_model =load_model(cross_validated=True, path=path)

        # Use testset to measure the accuracy of the model in the test data
        cv_preds = cv_model.bestModel.transform(testingData)

        metrics_list = ['mae', 'mse', 'rmse', 'r2']
        cv_metrics = []

        print("==========Performance Metrics on test set==========", )
        for m in metrics_list:
            res = eval_preds(cv_preds, m)
            cv_metrics.append((m, round(res, 2)))
            print(f'{m}: {res:.2f}')

        cv_res = spark.createDataFrame(cv_metrics, ["Metric", "Value"])
        metrics_path = metrics_dir / f"metrics_cv_{cv_ts}.parquet"
        cv_res.write.mode("overwrite").parquet(str(metrics_path))

    finally:
        spark.stop()