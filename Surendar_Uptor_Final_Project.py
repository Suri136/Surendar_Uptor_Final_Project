import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingRegressor


"""Load the dataset"""
df = pd.read_csv('Surendar_uptor_Final_project.csv')
print(df.head())

"""read only column names of the given dataset"""
df_column_names = df.columns
print(df_column_names)

""" print 0th row to 5th row (that means total of 6 rows) for the imdb_score """
row_data = df.loc[0:40,'imdb_score']
print(row_data)

"""print the unique data"""
finding_unique_categorical = df['imdb_score'].unique()
print(finding_unique_categorical)

"""print the nunique data"""
finding_unique_categorical_count = df['imdb_score'].nunique()
print(finding_unique_categorical_count)

"""print the category_value_count data"""
finding_unique_category_value_count = df.value_counts()
print(finding_unique_category_value_count)

"""print the unique_category_total_count data"""
finding_unique_category_total_count = df['imdb_score'].count()
print(finding_unique_category_total_count)

"""Preprocessing"""

"""Convert categorical columns to numerical values"""
le = LabelEncoder()
df['genres'] = le.fit_transform(df['genres'])
df['production_countries'] = le.fit_transform(df['production_countries'])
df['type'] = le.fit_transform(df['type'])
print("Categorical Columns Converted")
print(df.head())

"""handling missing the value """
df[['genres', 'production_countries', 'type', 'imdb_score']] = df[['genres', 'production_countries', 'type', 'imdb_score']].fillna(0)
print(df)

"""Print missing value counts"""
print(df[['genres', 'production_countries', 'type', 'imdb_score']].isnull().sum())
print(df)
print("After filling:")
nan_values_column = df[['genres', 'production_countries', 'type', 'imdb_score']].isnull().sum()
print(df.head())

"""Scale the data using StandardScaler"""
scaler = StandardScaler()
df[['genres', 'production_countries', 'type']] = scaler.fit_transform(df[['genres', 'production_countries', 'type']])
print("Data Scaled")
print(df.head())

"""Split data into features (X) and target variable(y)"""
X = df[['genres', 'production_countries', 'type']]
y = df['imdb_score']

"""Split the data into training and testing sets"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data Split into Training and Testing Sets:")
print("Training Features (X_train)")
print(X_train.head())
print("Training Target (y_train)")
print(y_train.head())
print("Testing Features (X_test)")
print(X_test.head())
print("Testing Target (y_test)")
print(y_test.head())

"""Train a random forest regressor"""
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
print("Regressor Trained")

"""Make predictions on the testing data"""
y_pred = rfr.predict(X_test)
print("Predictions Made")
print(y_pred)

"""Evaluate the model  performance"""
mse = mean_squared_error(y_test, y_pred)
print("Model Evaluated:")
print("Mean Squared Error", mse)

""" Unsupervised Learning """

"""Apply K-Means clustering"""
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df[['genres', 'production_countries', 'type']])
print("K-Means Clustering Applied:")

"""Predict the cluster labels"""
labels = kmeans.predict(df[['genres', 'production_countries', 'type']])
print("Cluster Labels Predicted:")
print(labels)

"""cluster labels to DataFrame"""
df['cluster'] = labels
print("Cluster Labels Added to DataFrame:")
print(df.head())

"""Visualize the clusters using PCA """
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(df[['genres', 'production_countries', 'type']])
print("PCA Applied:")
print(X_pca)

"""print the Visualize K-Mean clusters"""
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering')
plt.show()
print("Clusters Visualized")

"""Tune hyperparameters using RandomizedSearchCV"""
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

rfr = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rfr, param_distributions=param_grid, cv=5, n_iter=5, random_state=42)
random_search.fit(X_train, y_train)

"""Print the best hyperparameters and score"""
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

"""Train and evaluate the best model"""
best_rfr = RandomForestRegressor(**random_search.best_params_, random_state=42)
best_rfr.fit(X_train, y_train)
y_pred_best = best_rfr.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print("MSE with Best Model:", mse_best)

"""K-Means clustering"""
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df[['genres', 'production_countries', 'type']])
labels = kmeans.predict(df[['genres', 'production_countries', 'type']])
df['cluster'] = labels

""" print the PCA visualization"""
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(df[['genres', 'production_countries', 'type']])
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='BuGn')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering')
plt.show()

"""Identify categorical columns"""
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns", categorical_cols)

"""One-hot encode categorical variables"""
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_cols])
print("One-hot encoding done.")

"""Create new columns for encoded data"""
encoded_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
print("Encoded data created.")

"""Concatenate encoded data with numeric data"""
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
print("Data concatenated.")
#
"""Split data into features (X) and target variable (y)"""
X = df.drop('imdb_score', axis=1)
y = df['imdb_score']
print("Data split into X and y.")

""""Initialize Gradient Boosting model"""

gbr = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=7, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gb = gbr.predict(X_test)

"""Evaluate Gradient Boosting"""
mse_gb = mean_squared_error(y_test, y_pred_gb)
print("Gradient Boosting MSE:", mse_gb)

"""Use K-Means clustering to identify patterns in the movie datase"""
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(df[['genres', 'production_countries', 'type']])

"""Store labels in DataFrame"""
df['kmeans_cluster'] = labels_kmeans

"""Evaluate K-Means clustering performance"""
silhouette_kmeans = silhouette_score(df[['genres', 'production_countries', 'type']], labels_kmeans)
print("K-Means Silhouette Score:", silhouette_kmeans)

"""Use t-SNE to reduce the dimensionality of the dataset while preserving local structures"""

x= df[['genres', 'production_countries', 'type']]
y = df['imdb_score']

"""Preprocess categorical features"""
categorical_cols = ['genres', 'production_countries', 'type']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

"""Create a pipeline with preprocessing"""
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

"""Fit the pipeline to the data and transform"""
X_transformed = pipeline.fit_transform(X)

"""Apply t-SNE for dimensionality reduction"""
tsne = TSNE(n_components=2, random_state=42, perplexity=50)
X_tsne = tsne.fit_transform(X_transformed.toarray())

"""Plot the results"""
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.colorbar(label='IMDB Score')
plt.title('t-SNE Visualization')
plt.show()


print("Actual vs Predicted IMDb Scores:")

"""Loop to print actual vs predicted values"""
for actual, predicted in zip(y_test[:10], y_pred_gb[:10]):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

""" Visualize actual vs predicted values"""
plt.scatter(y_test, y_pred_gb)
plt.xlabel("Actual IMDb Scores")
plt.ylabel("Predicted IMDb Scores")
plt.title("Gradient Boosting Model Performance")
plt.show()

"""Print top 10 highest rated movies"""
print(df.columns)

"""Restore correct titles"""
original_df = pd.read_csv("Surendar_uptor_Final_project.csv")
df["title"] = original_df["title"]

print('Top 10 Highest Rated Movies')
print(df[['title', 'imdb_score']].sort_values(by='imdb_score', ascending=False).head(10))

"""print the first 10 rows of original data"""
print(df[['genres']].head())

"""Print top 10 highest rated movies"""
top_rated_movies = df[['title', 'imdb_score']].sort_values(by='imdb_score', ascending=False).head(10)

"""Visualize top 10 highest rated movies"""
plt.figure(figsize=(10,6))
sns.barplot(x="title", y="imdb_score", data=top_rated_movies)
plt.xlabel("Movie Title")
plt.ylabel("IMDb Score")
plt.title("Top 10 Highest Rated Movies")
plt.xticks(rotation=90)
plt.show()

"""Verify if 'imdb_score' column exists in the dataset"""
df = pd.read_csv("Surendar_uptor_Final_project.csv")

"""Ensure IMDb score column """
if 'imdb_score' not in df.columns:
    raise ValueError("Column 'imdb_score' not found in dataset!")

"""Sort movies by lowest IMDb scores"""
lowest_rated_movies = df[['title', 'imdb_score']].sort_values(by='imdb_score', ascending=True).head(10)

"""Print top 10 lowest rated movie"""
print("Top 10 Lowest Rated Movies:")
print(lowest_rated_movies)

"""Visualize lowest rated movies with bar chart"""
plt.figure(figsize=(12,6))
sns.barplot(x=lowest_rated_movies['imdb_score'], y=lowest_rated_movies['title'], color='skyblue')
plt.xlabel("IMDb Score")
plt.ylabel("Movie Title")
plt.title("Top 10 Lowest Rated Movies")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.show()

""" print clean movie genres"""
original_df = pd.read_csv("Surendar_uptor_Final_project.csv")
df['genres'] = original_df['genres']
print(df[['genres']].head())
df['genres'] = df['genres'].fillna('Unknown')

"""Count number of movies released per year"""
release_trends = df['release_year'].value_counts().sort_index()

"""Plot movie release trends overtime"""
plt.figure(figsize=(12,6))
release_trends.plot(kind='bar', color='skyblue')
plt.xlabel("Release Year")
plt.ylabel("Number of Movies Released")
plt.title("Movie Release Trends Over Time")
print('lowest moive')


"""Load the movie dataset from a CSV file"""
df = pd.read_csv("Surendar_uptor_Final_project.csv")

"""Ensure 'genres' and 'imdb_score'"""
df = df[['genres', 'imdb_score']].dropna()

"""Explode genres if stored as lists"""
df['genres'] = df['genres'].astype(str).str.strip("[]").str.replace("'", "").str.split(", ")
df = df.explode('genres')

"""Define a success threshold (IMDb > 7.5 is considered successful)"""
success_threshold = 7.5

"""Calculate success rate per genre"""
"""Top 10 success rates"""
df['successful'] = df['imdb_score'] >= success_threshold
genre_success_rate = df.groupby('genres')['successful'].mean().sort_values(ascending=False)[:10]

"""Reset index to ensure proper labeling"""
genre_success_rate = genre_success_rate.reset_index()
genre_success_rate.columns = ['Genre', 'Success Rate']

"""Load the movie dataset from a CSV file"""
df = pd.read_csv("Surendar_uptor_Final_project.csv")

"""Ensure 'imdb_votes' and 'imdb_score'"""
df = df[['imdb_votes', 'imdb_score']].dropna()

"""Calculate correlation"""
"""Pearson correlation"""
correlation = df.corr(method='pearson')
print("Correlation Matrix:\n", correlation)

"""Visualize IMDb votes vs score correlation"""
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['imdb_votes'], y=df['imdb_score'], alpha=0.7, color='blue')
plt.xlabel("IMDb Votes")
plt.ylabel("IMDb Score")
plt.title("IMDb Votes vs. IMDb Score (Positive Correlation)")
plt.show()

"""Fit regression line"""
slope, intercept, r_value, _, _ = stats.linregress(df['imdb_votes'], df['imdb_score'])

"""Load the movie dataset from a CSV file"""
df = pd.read_csv("Surendar_uptor_Final_project.csv")

""" print the 'imdb_id' column """
if 'imdb_id' not in df.columns:
    raise ValueError("Column 'imdb_id' not found in dataset!")

"""Count occurrences of each IMDb ID""" """Get top 10 IMDb IDs"""
imdb_counts = df['imdb_id'].value_counts().head(10)

"""Display results"""
"""Plot IMDb rating distribution barchart"""
plt.figure(figsize=(10,6))
sns.barplot(
     x=imdb_counts.index,
     y=imdb_counts.values,
     hue=imdb_counts.index,
     palette="Set2",
     legend=False
)

"""Display frequent IMDb IDs barchart"""
plt.xticks(rotation=90)
plt.xlabel("IMDb ID")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Frequent IMDb IDs")
plt.show()

"""Predict the future genres"""
future_genres = df.groupby('genres')['imdb_score'].mean().reset_index()
future_genres['future_imdb_score'] = future_genres['imdb_score'] * 1.1

"""Sort the genres by future IMDb score in descending order"""
future_genres = future_genres.sort_values(by='future_imdb_score', ascending=False)

""" print Top 10 future genres"""
top_future_genres = future_genres.head(10)

"""Print the top 10 future genres"""
print("Top 10 future genres")
print(top_future_genres)
