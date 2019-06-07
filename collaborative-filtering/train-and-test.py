# import ALS recommender function from pyspark ml library
from pyspark.ml.recommendation import ALS

# split the data into training and test datatset
train, test = indexed.randomSplit([0.75, 0.25])

# count number of records in train set
train.count()

# count number of records in test set
test.count()

# Training the recommender model using train datatset
rec = ALS(maxIter=10, regParam=0.01, userCol='userId', itemCol='title_new', ratingCol='rating', nonnegative=True,
          coldStartStrategy="drop")

# fit the model on train set
rec_model = rec.fit(train)

# making predictions on test set 
predicted_ratings = rec_model.transform(test)

# columns in predicted ratings dataframe
predicted_ratings.printSchema()

# predicted vs actual ratings for test set 
predicted_ratings.orderBy(rand()).show(10)