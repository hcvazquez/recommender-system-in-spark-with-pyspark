# import String indexer to convert string values to numeric values
from pyspark.ml.feature import StringIndexer, IndexToString

# creating string indexer to convert the movie title column values into numerical values
stringIndexer = StringIndexer(inputCol="title", outputCol="title_new")

# applying stringindexer object on dataframe movie title column
model = stringIndexer.fit(df)

# creating new dataframe with transformed values
indexed = model.transform(df)

# validate the numerical title values
indexed.show(10)

# number of times each numerical movie title has been rated 
indexed.groupBy('title_new').count().orderBy('count', ascending=False).show(10, False)