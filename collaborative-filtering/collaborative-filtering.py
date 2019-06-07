# Recommend top movies  which user might like 

# create dataset of all distinct movies 
unique_movies = indexed.select('title_new').distinct()

# number of unique movies
unique_movies.count()

# assigning alias name 'a' to unique movies df
a = unique_movies.alias('a')
user_id = 85

# creating another dataframe which contains already watched movie by active user 
watched_movies = indexed.filter(indexed['userId'] == user_id).select('title_new').distinct()

# number of movies already rated 
watched_movies.count()

# assigning alias name 'b' to watched movies df
b = watched_movies.alias('b')

# joining both tables on left join 
total_movies = a.join(b, a.title_new == b.title_new, how='left')
total_movies.show(10, False)

# selecting movies which active user is yet to rate or watch
remaining_movies = total_movies.where(col("b.title_new").isNull()).select(a.title_new).distinct()

# number of movies user is yet to rate 
remaining_movies.count()

# adding new column of user_Id of active useer to remaining movies df 
remaining_movies = remaining_movies.withColumn("userId", lit(int(user_id)))

remaining_movies.show(10, False)

# making recommendations using ALS recommender model and selecting only top 'n' movies
recommendations = rec_model.transform(remaining_movies).orderBy('prediction', ascending=False)
recommendations.show(5, False)

# converting title_new values back to movie titles
movie_title = IndexToString(inputCol="title_new", outputCol="title", labels=model.labels)
final_recommendations = movie_title.transform(recommendations)
final_recommendations.show(10, False)


# create function to recommend top 'n' movies to any particular user
def top_movies(user_id, n):
    """
    This function returns the top 'n' movies that user has not seen yet but might like 

    """
    # assigning alias name 'a' to unique movies df
    a = unique_movies.alias('a')

    # creating another dataframe which contains already watched movie by active user 
    watched_movies = indexed.filter(indexed['userId'] == user_id).select('title_new')

    # assigning alias name 'b' to watched movies df
    b = watched_movies.alias('b')

    # joining both tables on left join 
    total_movies = a.join(b, a.title_new == b.title_new, how='left')

    # selecting movies which active user is yet to rate or watch
    remaining_movies = total_movies.where(col("b.title_new").isNull()).select(a.title_new).distinct()

    # adding new column of user_Id of active useer to remaining movies df 
    remaining_movies = remaining_movies.withColumn("userId", lit(int(user_id)))

    # making recommendations using ALS recommender model and selecting only top 'n' movies
    recommendations = rec_model.transform(remaining_movies).orderBy('prediction', ascending=False).limit(n)

    # adding columns of movie titles in recommendations
    movie_title = IndexToString(inputCol="title_new", outputCol="title", labels=model.labels)
    final_recommendations = movie_title.transform(recommendations)

    # return the recommendations to active user
    return final_recommendations.show(n, False)

top_movies(85, 10)
