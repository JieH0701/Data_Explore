import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

books = pd.read_csv('books/books.csv', error_bad_lines=False)
books.drop_duplicates(subset='original_title', keep=False, inplace=True)
ratings = pd.read_csv('books/ratings.csv')
ratings.drop_duplicates(subset=["user_id", "book_id"], keep=False, inplace=True)

books_col = books[['book_id', 'original_title']]
books_col.dropna()
books_col = books_col[books_col['original_title'].str.isnumeric().fillna(False)]
# pivot ratings into books features
df_book_features = ratings.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
mat_book_features = csr_matrix(df_book_features.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20, n_jobs=-1)

num_users = len(ratings.user_id.unique())
num_items = len(ratings.book_id.unique())
print('There are {} unique users and {} unique books in this data set'.format(num_users, num_items))
ratings = ratings.dropna()

df_ratings_cnt = pd.DataFrame(ratings.groupby('rating').size(), columns=['count'])
df_ratings_cnt.head(10)

df_ratings_cnt = df_ratings_cnt[['count']].reset_index().rename(columns={'index': 'rating score'})

fig, ax = plt.subplots()
plt.bar(df_ratings_cnt['rating score'], df_ratings_cnt['count'])
plt.xticks(df_ratings_cnt['rating score'])
ax.set_title('Count for Each Rating Score')
ax.set_xlabel('Books Rating Score')
ax.set_ylabel('Number of Ratings')

df_books_cnt = pd.DataFrame(ratings.groupby('book_id').size(), columns=['count'])
df_books_cnt.head()

# now we need to take only books that have been rated atleast 60 times
# to get some idea of the reactions of users towards it

popularity_thres = 60

popular_books = df_books_cnt[df_books_cnt['count'].ge(popularity_thres)].index
df_ratings_drop = ratings[ratings.book_id.isin(popular_books)]
print('shape of original ratings data: ', ratings.shape)
print('shape of ratings data after dropping unpopular books: ', df_ratings_drop.shape)

# get number of ratings given by every user
df_users_cnt = pd.DataFrame(df_ratings_drop.groupby('user_id').size(), columns=['count'])
df_users_cnt.head()

ratings_thres = 50
active_users = df_users_cnt[df_users_cnt['count'].ge(ratings_thres)].index
df_ratings_drop_users = df_ratings_drop[df_ratings_drop.user_id.isin(active_users)]
print('shape of original ratings data: ', ratings.shape)
print('shape of ratings data after dropping both unpopular books and inactive users: ', df_ratings_drop_users.shape)

book_user_mat = df_ratings_drop_users.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
book_user_mat_sparse = csr_matrix(book_user_mat.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

# fit
model_knn.fit(book_user_mat_sparse)


def fuzzy_matching(mapper, fav_book, verbose=True):
    match_tuple = []
    for title, idx in mapper.items():
        if type(title) == str:
            ratio = fuzz.ratio(title.lower(), fav_book.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2], reverse=True)
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


def make_recommendation(model_knn, data, mapper, fav_book, n_recommendations):
    model_knn.fit(data)
    # get input movie index
    print('You have input book:', fav_book)
    idx = fuzzy_matching(mapper, fav_book)

    print('Recommendation system starting to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)

    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1]
                            , reverse=True)
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_book))
    rec = []
    for i, (idx, dist) in enumerate(raw_recommends):
        if idx not in reverse_mapper.keys():
            continue
        print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_mapper[idx], dist))
        rec.append(reverse_mapper[idx])
    return rec


my_favorite = 'Factfullness'
indices = pd.Series(books_col.index, index=books_col['original_title'])

make_recommendation(
    model_knn=model_knn,
    data=book_user_mat_sparse,
    fav_book='The God Delusion',
    mapper=indices,
    n_recommendations=10)

make_recommendation(
    model_knn=model_knn,
    data=book_user_mat_sparse,
    fav_book='Homo Deus',
    mapper=indices,
    n_recommendations=10)
