# importing Regression Evaluator to measure RMSE
from pyspark.ml.evaluation import RegressionEvaluator

# create Regressor evaluator object for measuring accuracy
evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction', labelCol='rating')

# apply the RE on predictions dataframe to calculate RMSE
rmse = evaluator.evaluate(predicted_ratings)

# print RMSE error
print(rmse)