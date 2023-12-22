import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
from collections import defaultdict
from tqdm import tqdm
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display
from statistics import median
import scipy.stats as stats
import math

"""Function to create the ranking based on the average movie revenue and rating"""
def get_rating_stand(df1):
    # Filter years to include only films from 1980 to 2020
    df2 = df1[(df1['Movie_release'] >= 1980) & (df1['Movie_release'] <= 2020)]

    # Step 1: Mapping DataFrame for 'Actor_pairs' to 'Actor1', 'Actor2', and 'Genre'
    actor_pairs_mapping = df2[['Actor_pairs', 'Actor1', 'Actor2', 'Genre']].drop_duplicates()

    # Step 2: Grouping by 'Actor_pairs' and calculating metrics
    grouped_df = df2.groupby('Actor_pairs').agg(
        Average_Movie_revenue=pd.NamedAgg(column='Movie_revenue', aggfunc='mean'),
        Average_Movie_rating=pd.NamedAgg(column='Movie_rating', aggfunc='mean'),
        Count=pd.NamedAgg(column='Movie_name', aggfunc='count')
    ).reset_index()

    # Step 3: Merging aggregated DataFrame with the mapping DataFrame
    final_df = pd.merge(grouped_df, actor_pairs_mapping, on='Actor_pairs')

    # Filter to only keep real duos
    duos = final_df[final_df['Count'] >= 3]

    # Standardizing
    duos_standardized = duos.copy()
    standard_scaler = MinMaxScaler()
    cols_to_normalize = ['Average_Movie_revenue', 'Average_Movie_rating']
    duos_standardized[cols_to_normalize] = standard_scaler.fit_transform(duos_standardized[cols_to_normalize])

    # Round down the revenue
    duos_standardized['Average_Movie_revenue'] = duos_standardized['Average_Movie_revenue'].apply(lambda x: np.floor(x / 0.05) * 0.05)

    # Sort and rank
    rating_stand = duos_standardized.sort_values(by=["Average_Movie_rating", "Average_Movie_revenue"], ascending=False)
    rating_stand.reset_index(drop=True, inplace=True)
    rating_stand['rank'] = rating_stand.index + 1

    # Adjusting ranks for ties
    for i in range(1, len(rating_stand)):
        if (rating_stand.loc[i, 'Average_Movie_revenue'] == rating_stand.loc[i - 1, 'Average_Movie_revenue']) and (rating_stand.loc[i, 'Average_Movie_rating'] == rating_stand.loc[i - 1, 'Average_Movie_rating']):
            rating_stand.loc[i, 'rank'] = rating_stand.loc[i - 1, 'rank']

    # Rank ratio and color transformation
    length = len(rating_stand)
    rating_stand['rank_ratio'] = (length - (rating_stand['rank'] - 1)) / length
    rating_stand['Color'] = rating_stand['rank_ratio'].apply(lambda x: (0, (x - 0.5) * 2, 0.3) if x >= 0.5 else (np.abs((x - 0.5) * 2), 0, 0.3))

    return rating_stand


# Function to create the genre distribution for each cluster based on movies
def process_clusters_and_genres(df, large_clusters, rating_stand):
    actor_movies = df.groupby('Actor_name')['Movie_name'].apply(set)
    movie_genres = df.drop_duplicates('Movie_name').set_index('Movie_name')['Main_genre']
    cluster_info = defaultdict(lambda: {'movie_genre_counts': defaultdict(int), 'total_movie_count': 0, 'average_rank': 0})

    for cluster_id, actors in tqdm(large_clusters.items(), desc='Processing clusters'):
        movies_counted = set()
        for actor1, actor2 in itertools.combinations(actors, 2):
            movies_together = actor_movies.get(actor1, set()).intersection(actor_movies.get(actor2, set()))
            unique_movies_together = movies_together - movies_counted
            movies_counted.update(unique_movies_together)

            for movie in unique_movies_together:
                genre = movie_genres.get(movie)
                if genre:
                    cluster_info[cluster_id]['movie_genre_counts'][genre] += 1

        total_movie_count = sum(cluster_info[cluster_id]['movie_genre_counts'].values())
        cluster_info[cluster_id]['total_movie_count'] = total_movie_count

        cluster_rows = rating_stand[rating_stand['Actor1'].isin(actors) | rating_stand['Actor2'].isin(actors)]
        cluster_info[cluster_id]['average_rank'] = cluster_rows['rank'].mean() if not cluster_rows.empty else None

    sorted_clusters = sorted(cluster_info.items(), key=lambda item: item[1]['average_rank'] or float('inf'))

    for cluster_id, info in sorted_clusters:
        total_movies = info['total_movie_count']
        if total_movies > 0:
            info['movie_genre_counts'] = {
                genre: count / total_movies
                for genre, count in info['movie_genre_counts'].items()
                if count / total_movies >= 0.02
            }
            info['movie_genre_counts']['Other Genres'] = sum(
                count for count in info['movie_genre_counts'].values()
                if count / total_movies < 0.02
            )

        print(f"Cluster {cluster_id}:")
        print(f"  Average Rank: {info['average_rank']}")
        for genre, ratio in info['movie_genre_counts'].items():
            print(f"  {genre}: {ratio:.2%}")
        print()
    
    return cluster_info


# Function to create the plots for movies
def create_and_display_plots(cluster_info):
    sorted_cluster_ids = sorted(cluster_info, key=lambda x: cluster_info[x]['average_rank'] or float('inf'))
    cluster_ids = [f"Cluster {cluster_id}" for cluster_id in sorted_cluster_ids]
    average_ranks = [round(cluster_info[cluster_id]['average_rank']) for cluster_id in sorted_cluster_ids]

    bar_chart = go.Figure(data=[go.Bar(x=cluster_ids, y=average_ranks, name='Average Rank')])
    bar_chart.update_layout(clickmode='event+select')
    pie_chart = go.FigureWidget()

    def update_pie_chart(change):
        cluster_id_str = change['new']
        if cluster_id_str:
            cluster_id = int(cluster_id_str.split()[1])
            genres = list(cluster_info[cluster_id]['movie_genre_counts'].keys())
            counts = list(cluster_info[cluster_id]['movie_genre_counts'].values())

            total = sum(counts)
            ratios = [count / total for count in counts]

            pie_chart.data = []
            pie_chart.add_trace(go.Pie(labels=genres, values=ratios))

    bar_select = widgets.Dropdown(options=cluster_ids, description='Select Cluster:', disabled=False)
    bar_select.observe(update_pie_chart, names='value')

    display(bar_select)
    display(bar_chart)
    display(pie_chart)


# Function to perform the statistical tests for movies
def perform_t_tests(cluster_info):
    t_test_results = []
    all_genres = set()
    for info in cluster_info.values():
        all_genres.update(info['movie_genre_counts'].keys())

    for genre in all_genres:
        genre_ratios = []
        average_ranks = []

        for cluster_id, info in cluster_info.items():
            if info['total_movie_count'] > 0 and genre in info['movie_genre_counts']:
                genre_ratio = info['movie_genre_counts'][genre]
                genre_ratios.append(genre_ratio)
                average_ranks.append(info['average_rank'])

        median_ratio = median(genre_ratios)
        high_ratio_ranks = [rank for ratio, rank in zip(genre_ratios, average_ranks) if ratio > median_ratio]
        low_ratio_ranks = [rank for ratio, rank in zip(genre_ratios, average_ranks) if ratio <= median_ratio]

        t_stat, p_value = stats.ttest_ind(high_ratio_ranks, low_ratio_ranks, nan_policy='omit')
        if not math.isnan(t_stat) and not math.isnan(p_value):
            t_test_results.append((genre, t_stat, p_value))

    for genre, t_stat, p_value in t_test_results:
        print(f"Genre: {genre}")
        print(f"  T-statistic: {t_stat}, P-value: {p_value}")
        print()

    return t_test_results


# Function to create the genre distribution for each cluster based on actors
def process_cluster_genre_counts(df, large_clusters, rating_stand):
    cluster_info = defaultdict(lambda: {'genre_counts': defaultdict(int), 'total_genre_count': 0, 'average_rank': 0})

    for cluster_id, nodes in large_clusters.items():
        for actor in nodes:
            main_genres = df[df['Actor_name'] == actor]['Actor_main_genre'].unique()
            for genre in main_genres:
                cluster_info[cluster_id]['genre_counts'][genre] += 1
                cluster_info[cluster_id]['total_genre_count'] += 1

        cluster_rows = rating_stand[(rating_stand['Actor1'].isin(nodes)) | (rating_stand['Actor2'].isin(nodes))]
        cluster_info[cluster_id]['average_rank'] = cluster_rows['rank'].mean() if not cluster_rows.empty else None

    sorted_clusters = sorted(cluster_info.items(), key=lambda item: item[1]['average_rank'] if item[1]['average_rank'] is not None else float('inf'))

    for cluster_id, info in sorted_clusters:
        print(f"Cluster {cluster_id}:")
        print(f"  Average Rank: {info['average_rank']}")
        if info['total_genre_count'] > 0:
            for genre, count in info['genre_counts'].items():
                ratio = count / info['total_genre_count']
                print(f"  {genre}: {ratio:.2f}")
        print()

    return cluster_info


# Function to create the plots for actors
def create_interactive_plots(cluster_info):
    sorted_cluster_ids = sorted(cluster_info, key=lambda x: cluster_info[x]['average_rank'])
    cluster_ids = [f"Cluster {cluster_id}" for cluster_id in sorted_cluster_ids]
    average_ranks = [round(cluster_info[cluster_id]['average_rank']) for cluster_id in sorted_cluster_ids]

    bar_chart = go.Figure(data=[go.Bar(x=cluster_ids, y=average_ranks, name='Average Rank')])
    pie_chart = go.FigureWidget()

    def update_pie_chart(change):
        cluster_id_str = change['new']
        if cluster_id_str:
            cluster_id = int(cluster_id_str.split()[1])
            genres = list(cluster_info[cluster_id]['genre_counts'].keys())
            counts = list(cluster_info[cluster_id]['genre_counts'].values())
            pie_chart.data = []
            pie_chart.add_trace(go.Pie(labels=genres, values=counts))

    cluster_select = widgets.Dropdown(options=cluster_ids, description='Select Cluster:', disabled=False)
    cluster_select.observe(update_pie_chart, names='value')

    display(cluster_select)
    display(bar_chart)
    display(pie_chart)


# Function to perform the statistical tests for actors
def perform_genre_t_tests(cluster_info):
    t_test_results = []
    all_genres = set()
    for info in cluster_info.values():
        all_genres.update(info['genre_counts'].keys())

    for genre in all_genres:
        genre_ratios = []
        average_ranks = []
        for cluster_id, info in cluster_info.items():
            if info['total_genre_count'] > 0 and genre in info['genre_counts']:
                genre_ratio = info['genre_counts'][genre] / info['total_genre_count']
                genre_ratios.append(genre_ratio)
                average_ranks.append(info['average_rank'])

        median_ratio = median(genre_ratios)
        high_ratio_ranks = [rank for ratio, rank in zip(genre_ratios, average_ranks) if ratio > median_ratio]
        low_ratio_ranks = [rank for ratio, rank in zip(genre_ratios, average_ranks) if ratio <= median_ratio]

        t_stat, p_value = stats.ttest_ind(high_ratio_ranks, low_ratio_ranks, nan_policy='omit')
        if not math.isnan(t_stat) and not math.isnan(p_value):
            t_test_results.append((genre, t_stat, p_value))

    for genre, t_stat, p_value in t_test_results:
        print(f"Genre: {genre}")
        print(f"  T-statistic: {t_stat}, P-value: {p_value}")
        print()

    return t_test_results
