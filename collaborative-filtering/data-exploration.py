# validate the shape of the data 
print((df.count(), len(df.columns)))

# check columns in dataframe
df.printSchema()

# validate few rows of dataframe in random order
df.orderBy(rand()).show(10, False)

# check number of ratings by each user
df.groupBy('userId').count().orderBy('count', ascending=False).show(10, False)

# check number of ratings by each user
df.groupBy('userId').count().orderBy('count', ascending=True).show(10, False)

# number of times movie been rated 
df.groupBy('title').count().orderBy('count', ascending=False).show(10, False)
df.groupBy('title').count().orderBy('count', ascending=True).show(10, False)