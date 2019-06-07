# import and create sparksession object
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('rc').getOrCreate()

# import the required functions and libraries
from pyspark.sql.functions import *

# Convert csv file to Spark DataFrame (Databricks version)
def loadDataFrame(fileName, fileSchema):
    return (spark.read.format("csv").schema(fileSchema).option("header", "true").option("mode", "DROPMALFORMED").csv(
        "/FileStore/tables/%s" % (fileName)))


from pyspark.sql.types import *

movieRatingSchema = StructType([StructField("userId", IntegerType(), True), StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True), StructField("timestamp", StringType(), True)])
movieSchema = StructType([StructField("movieId", IntegerType(), True), StructField("title", StringType(), True),
    StructField("genres", StringType(), True)])
MovieRatingsDF = loadDataFrame("ratings.csv", movieRatingSchema).cache()
MoviesDF = loadDataFrame("movies.csv", movieSchema).cache()

# load the dataset and create sprk dataframe
df = MovieRatingsDF.join(MoviesDF, 'movieId').select(['userId', 'title', 'rating'])