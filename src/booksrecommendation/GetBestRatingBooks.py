import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# First step, remove the duplicates
books = pd.read_csv('books/books.csv', error_bad_lines=False)
books.drop_duplicates(subset='original_title', keep=False, inplace=True)

ratings = pd.read_csv('books/ratings.csv')
ratings = ratings.sort_values("user_id")
ratings.drop_duplicates(subset=["user_id", "book_id"], keep=False, inplace=True)

tags = pd.read_csv('books/book_tags.csv')
tags.drop_duplicates(subset=['tag_id', 'goodreads_book_id'], keep=False, inplace=True)

btags = pd.read_csv('books/tags.csv')
btags.drop_duplicates(subset='tag_id', keep=False, inplace=True)

rows = ['title', 'authors', 'average_rating', 'ratings_count']
top_rated = books.sort_values('average_rating', ascending=False)
top_rated_10 = top_rated.head(10)

most_rated = books.sort_values(by='ratings_count', ascending=False)
most_rated_10 = most_rated[rows].head(10)
most_rated_10 = most_rated_10.set_index('title')

displ = (most_rated_10[rows])
displ.set_index('title', inplace=True)

# Most Common Rating Values
plt.figure(figsize=(16, 8))
sns.distplot(a=books['average_rating'], kide=True, color='r')
# Therefore, the most common rating is somewhere between 3.5 to 4.

top_authors = books[['authors', 'average_rating']]
top_authors = top_authors.groupby(['authors']).mean().round({'average_rating': 2}).reset_index()

top_authors = top_authors[['authors', 'average_rating']].sort_values('average_rating', ascending=False)

fig = px.bar(top_authors.head(20), x='authors', y='average_rating', color='average_rating')

genres = ["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics", "Comics", "Contemporary",
          "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction", "Gay and Lesbian", "Graphic Novels",
          "Historical Fiction", "History", "Horror", "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery",
          "Nonfiction", "Paranormal", "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science",
          "Science Fiction", "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]
for i in range(len(genres)):
    genres[i] = genres[i].lower()

joint_tags = pd.merge(tags, btags, left_on='tag_id', right_on='tag_id')[['goodreads_book_id', 'tag_name']]

nonfiction = joint_tags[joint_tags['tag_name'].str.contains('nonfiction', na=False)]

nonfiction[['tag_name']].unique()

books_filter = books[['title', 'best_book_id', 'average_rating']]
fictions_books = pd.merge(books_filter, nonfiction, left_on='best_book_id', right_on='goodreads_book_id')
fictions_books = fictions_books[['title', 'tag_name', 'average_rating']]

tags_count = joint_tags.groupby('tag_name').count().sort_values(by='count', ascending=False)
new_tags = tags_count[tags_count.index.isin(genres)].reset_index()


# K-Means
