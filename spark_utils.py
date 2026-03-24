from datetime import datetime
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel

def create_model(model=None, features_col=None, label_col=None, **kwargs):

    model = model(featuresCol=features_col, labelCol=label_col, kwargs=kwargs)

    return model

def vect_assembler(features=None, output='features'):

    assembler = VectorAssembler(inputCols=features, outputCol=output)

    return assembler

def create_pipeline(model=None, stages=None):
    stages.append(model)
    pipeline = Pipeline(stages=stages)

    return pipeline


def evaluators(label=None, pred_col=None, metrics_list=None):
    evaluators_dict = {}
    for metric in metrics_list:
        evaluator = RegressionEvaluator(labelCol=label, predictionCol=pred_col, metricName=metric)
        evaluators_dict[metric] = evaluator
    return evaluators_dict


def train(train_data, val_data, stages,  model=None, evaluators=None, path=None):

    metrics = {}
    model_name = model.__class__.__name__
    print(f'.........Training {model_name}.........')
    pipeline = create_pipeline(model=model, stages=stages)
    fit_pipeline = pipeline.fit(train_data)
    predictions = fit_pipeline.transform(val_data)
    model_eval = evaluation(evaluators, predictions)
    metrics[model_name] = model_eval
    model_path = f'{path}/{model_name}'
    fit_pipeline.write().overwrite().save(f'{model_path}')

    return metrics, model_path, model_name


def evaluation(evaluators=None, predictions=None):

    metrics = {}

    for metric, evaluator in evaluators.items():
        result = evaluator.evaluate(predictions)
        metrics[metric] = round(result, 2)
        print(f'{metric}: {result:.2f}')

    return metrics


def cross_validation(estimator=None, grid_params=None, folds=5, data=None, **kwargs):

    # Prepare the cross validator
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid_params, numFolds=folds)

    # Fit the model
    model = cv.fit(data)
    print(f'Model {model}')
    print(f'Model metrics: {model.avgMetrics}')

    ts =datetime.timestamp(datetime.now())
    path = f"{kwargs['path']}_{ts}"

    model.write().save(path)

    return model.bestModel, path


def load_model(cross_validated=False, path=None):

    if cross_validated:

        best_model = CrossValidatorModel.read().load(path)
    else:
        best_model = PipelineModel.load(path)

    return best_model


def evaluate_cross_validation(cross_validate=True, path=None, evaluators=None, test_data=None):

    model = load_model(cross_validate, path)
    predictions = model.transform(test_data)
    metrics = evaluation(evaluators, predictions)

    return metrics


