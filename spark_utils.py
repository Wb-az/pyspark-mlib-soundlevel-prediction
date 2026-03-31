from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidatorModel


def build_spark():
    conf = SparkConf().set("spark.executor.memory", "2g")
    spark = SparkSession.builder.config(conf=conf).appName("cv_regression").getOrCreate()
    spark.sparkContext.setLogLevel("error")
    return spark


def built_pipeline(data, regressor):
    assembler = VectorAssembler(inputCols=data.columns[0:-1], outputCol='features')
    scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
    pipeline = Pipeline(stages=[assembler, scaler, regressor])
    return pipeline


def regression_evaluator(metric="rmse"):
    return RegressionEvaluator(
        predictionCol="prediction",
        labelCol="SoundLevelDecibels",
        metricName=metric,
    )


def eval_preds(preds, metric):
    evaluator = regression_evaluator(metric)
    return evaluator.evaluate(preds)


def multimetric_evaluator(preds, metrics_list):

    metrics_dict = {}
    for m in metrics_list:
        res = eval_preds(preds, m)
        metrics_dict[m] = round(res, 2)
        print(f'{m}: {res:.2f}')

    return metrics_dict


def train(train_data, val_data, model=None, path=None, metrics_list=None):

    model_name = model.__class__.__name__
    print(f'.........Training {model_name}.........')
    pipeline = built_pipeline(train_data, regressor=model)
    fit_pipeline = pipeline.fit(train_data)
    preds = fit_pipeline.transform(val_data)
    metrics = multimetric_evaluator(preds, metrics_list)

    model_path = f'{path}/{model_name}'
    fit_pipeline.write().overwrite().save(f'{model_path}')

    return metrics, model_path, model_name


def load_model(cross_validated=False, path=None):

    if cross_validated:

        model = CrossValidatorModel.read().load(path)
    else:
        model = PipelineModel.load(path)

    return model


