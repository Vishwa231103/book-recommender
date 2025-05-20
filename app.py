import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# --- Load and prepare data ---
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
books = pd.read_csv('BX_Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication',
                 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

active_users = ratings['User-ID'].value_counts()
active_users = active_users[active_users >= 100].index
ratings_filtered = ratings[ratings['User-ID'].isin(active_users)]

popular_books = ratings_filtered['ISBN'].value_counts()
popular_books = popular_books[popular_books >= 50].index
ratings_filtered = ratings_filtered[ratings_filtered['ISBN'].isin(popular_books)]

ratings_filtered['ISBN'] = ratings_filtered['ISBN'].str.strip()
books['ISBN'] = books['ISBN'].str.strip()
ratings_with_books = ratings_filtered.merge(books, on='ISBN')

# --- Pivot and train model ---
user_item_pivot = ratings_with_books.pivot_table(index='User-ID',
                                                 columns='Book-Title',
                                                 values='Book-Rating',
                                                 fill_value=0)

user_item_sparse = csr_matrix(user_item_pivot.values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_item_sparse.T)

# --- Recommendation function ---
def recommend_books(book_title, n_recommendations=5):
    if book_title not in user_item_pivot.columns:
        return f"❌ '{book_title}' not found in the book list."
    
    book_vector = user_item_pivot[book_title].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(book_vector, n_neighbors=n_recommendations + 1)

    all_titles = list(user_item_pivot.columns)
    recommended_titles = [all_titles[i] for i in indices.flatten() if all_titles[i] != book_title]
    
    return recommended_titles[:n_recommendations]

# --- Streamlit UI ---
st.title("📚 Book Recommendation System")
book_list = list(user_item_pivot.columns)

selected_book = st.selectbox("Choose a book to get recommendations:", book_list)

if st.button("Get Recommendations"):
    recommendations = recommend_books(selected_book)
    st.subheader("📖 Recommended Books:")
    for i, title in enumerate(recommendations, start=1):
        st.write(f"{i}. {title}")
