import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix

def get_categorial_features(df, columns):
    """Transform categorial features into one-hot encoding type.
        
        Parameters
        ----------
        df: {pandas-dataframe} dataframe which contains features should be
        transformed
        
        columns: {python-list, numpy-array} name of columns which are categorial
        and had to be represent in one-hot-encoding style. They must be a subset of
        df.columns
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe with transformed columns
    """
    
    cur_df = pd.DataFrame(df.index)
    label = LabelEncoder()
    onehot = OneHotEncoder()
    
    for feature in columns:
        label.fit(df[feature])
        improved_features = label.transform(df[feature])
        onehot.fit(improved_features.reshape(-1, 1))
        cur_df = cur_df.join(pd.DataFrame(onehot.transform(improved_features.reshape(-1, 1))\
                                          .toarray(), columns=feature + '_' + label.classes_))
    cur_df.index = cur_df[cur_df.columns[0]]
    return cur_df.drop(cur_df.columns[0], axis=1)

def get_genres(df, genres):
    """Return genres from dataframe like boolean dataframe.
        
        Parameters
        ----------
        df: {pandas-dataframe} dataframe with data and genres column
        
        genres: {numpy-array, python-list} list with genres which will be a
        columns name.
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe boolean like represents genres of
        film.
    """
    
    cur_df = pd.DataFrame(index=df.index)
    for i in genres:
        cur_df[i] = 0
    #     cur_df = df.reindex(columns=np.append(df.columns, genres), fill_value=0)
    labels = df.apply(label_genre, axis=1)
    for num, row_label in enumerate(labels):
        for one_label in row_label:
            cur_df.loc[cur_df.index[num], one_label] = 1
    return cur_df

def label_genre(row):
    """Get row and separate it genres column by '|' letter.
        
        Parameters
        ----------
        row: {pandas-dataframe} dataframe consisting from one row which contain 'genres'
        column.
        
        Returns
        -------
        genres: {python-list} list with genres from cell.
    """
    genres = row['genres'].split('|')
    return genres

def to_adjacency_df(df, users_id, movies_id):
    """From edge dataframe creates adjacency dataframe with shape {user * movies}.
        Cell (i, j) contains rating from user_i to movie_j.
        
        Parameters
        ----------
        df: {pandas-dataframe} dataframe which contains edge list with ratings
        which we use like weights.
        
        users_id: {list, numpy-array} users information about we have in df
        
        movies_id: {list, numpy-array} movies information about we have in df
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe represented as adjacency matrix
    """
    
    as_matrix = df.as_matrix()
    shape = tuple(as_matrix.max(axis=0)[:2] + 1)
    coo = coo_matrix((as_matrix[:, 2], (as_matrix[:, 0], as_matrix[:, 1])), \
                     shape=shape, dtype=as_matrix.dtype)
    coo = coo.todense()
    cur_df = pd.DataFrame(coo[1:, 1:], index=range(1, coo.shape[0]), \
                                           columns=range(1, coo.shape[1]))
    cur_df = cur_df.loc[users_id, movies_id]
    return cur_df

def average_based_on_user_movie(df, name='avg', axis=0):
    """Counting sum across axis and non zero values and return dataframe
        containing average values with the same index.
        
        Parameters
        ----------
        df: {pandas-dataframe}
        
        name: {string-like} name of column with average values in returning
        dataframe
        
        axis: {int-like} 0 or 1 integer which mean axis summing by
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe with average values across axis
    """
    non_zero = np.count_nonzero(df.as_matrix(), axis=axis)
    df = df.dropna(axis=axis)
    avg = df.sum(axis=axis) / non_zero
    avg[avg == np.nan] = 0
    cur_df = pd.DataFrame(avg, columns=[name])
    return cur_df.dropna()

def join_dataframes(main_df, df, id_name):
    """Join dataframes by id_name. In main_df id_name is a column and in
        df it is an index.
        
        Parameters
        ----------
        main_df: {pandas-dataframe} dataframe is a base in which new features
        will be added
        
        df: {pandas-dataframe} dataframe with new features
        
        id_name: {string-like} string linking two dataframes
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe with new features
    """
    cur_df = main_df.copy()
    
    shape = []
    for ind in df.index:
        shape.append(cur_df[main_df[id_name] == ind].shape[0])
    
    for i in df.columns:
        cur_df[i] = np.repeat(df[i], shape).as_matrix()
    return cur_df

def join_scalar_product(main_df, df):
    """Join feature from avearges values.
        
        Parameters
        ----------
        main_df: {pandas-dataframe} dataframe is a base in which new features
        will be added
        
        df: {pandas-dataframe} dataframe with new feature with size users * movies
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe with new feature
    """
    new_column = np.array([])
    for ind in df.index:
        sub_table = main_df[main_df['user_id'] == ind]
        new_column = np.append(new_column, df.loc[ind, sub_table['movie_id']].as_matrix())
    main_df['product_feature'] = new_column
    return main_df

def average_based_on_genres(df, index, columns):
    """Counting average score for each genre.
        
        Parameters
        ----------
        df: {pandas-dataframe} dataframe with genres
        
        index: {array-like} users indexes
        
        columns: {array-like} numpy array with genres names
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe with average score for genres
    """
    
    cur_df = pd.DataFrame(index=index, columns=columns)
    cur_df = cur_df.fillna(0)
    for user in index:
        sub_table = df[df.user_id == user]
        ratings = sub_table.rating[:, np.newaxis] * sub_table[columns]
        summed_rates = np.sum(ratings, axis=0).as_matrix()
        non_zero = np.count_nonzero(sub_table[columns], axis=0)
        ans = np.zeros(len(summed_rates))
        ans[np.where(summed_rates != 0)] = summed_rates[np.where(summed_rates != 0)] / \
            non_zero[np.where(summed_rates != 0)]
        cur_df.loc[user] = ans
    
    return cur_df


def add_feature_for_one_df(df, users_ind, movies_ind, df_users, df_movies):
    """Adding all features which we need by task for one dataframe
        
        Parameters
        ----------
        df_list: {pandas-dataframe} dataframe with ratings splited into train and
        test samples
        
        users_ind: {array-like} users indexes by in splited part
        
        movies_ind: {array-like} movies indexes by in splited part
        
        df_movies: {pandas-dataframe} dataframe with movies and their genres and names
        
        df_user: {pandas-dataframe} dataframe with users and some information about them
        
        Returns
        -------
        df_result: {pandas-dataframe} dataframe with all features that we need
    """
    
    genres = ['Action', 'Adventure', 'Animation', 'Children\'s', \
              'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', \
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', \
              'Sci-Fi', 'Thriller', 'War', 'Western']
    features_categorial = ['age', 'gender', 'occupation']
    df_adjacency = to_adjacency_df(df, users_ind, movies_ind)
    df_movies_avg = average_based_on_user_movie(df_adjacency, 'movies_avg')
    df_users_avg = average_based_on_user_movie(df_adjacency, 'user_avg', axis=1)
    df_ohe_movie_genre = get_genres(df_movies.loc[movies_ind], genres)
    df_join_genres = join_dataframes(df, df_ohe_movie_genre, 'movie_id')
    df_genres_avg = average_based_on_genres(df_join_genres, users_ind, genres)
    scalar_product_on_genres_avg = np.dot(df_genres_avg, df_ohe_movie_genre.T) / \
            np.sum(df_ohe_movie_genre, axis=1)[np.newaxis, :]
    df_scalar_product = pd.DataFrame(scalar_product_on_genres_avg, \
                                         index=users_ind, columns=movies_ind)
    df_ohe_user_some = get_categorial_features(df_users, features_categorial)
    df_result = join_dataframes(df_join_genres, df_movies_avg, 'movie_id')
    df_result = join_dataframes(df_result, df_users_avg, 'user_id')
    df_result = join_dataframes(df_result, df_ohe_user_some, 'user_id')
    df_result = join_scalar_product(df_result, df_scalar_product)
    df_result['const'] = np.ones([df_result.shape[0]])
    return df_result


def add_my_features(df, df_users, df_movies):
    """Adding features that I create for one data frame
        
        Parameters
        ----------
        df_list: {pandas-dataframe} dataframe with ratings splited into train and
        test samples
        
        df_movies: {pandas-dataframe} dataframe with movies and their genres and names
        
        df_user: {pandas-dataframe} dataframe with users and some information about them
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe with all features that we need
    """
    zip_code = df_users.zip_code.as_matrix()
    for i in range(zip_code.shape[0]):
        zip_code[i] = zip_code[i][:5]
    zip_code = zip_code.astype(int)
    zip_code = np.divide(zip_code, 10000).astype(int).astype(str)
    df_zip = get_categorial_features(pd.DataFrame(zip_code, columns=['code'], \
                                                  index=df_users.index), ['code'])

    title = df_movies.title.copy().as_matrix()
    
    for i in range(title.shape[0]):
        title[i] = title[i][-5:-1]
    df_name = pd.DataFrame(title.astype(int), columns=['name'], index=df_movies.index)

    cur_df = df.copy()
    cur_df = join_dataframes(cur_df, df_zip, 'user_id')
    cur_df = join_dataframes(cur_df, df_name, 'movie_id')
    return cur_df

def add_my_features_to_folds(df_list, df_users, df_movies):
    """Adding features that I create for list of dataframe
        
        Parameters
        ----------
        df_list: {pandas-dataframe} dataframe with ratings splited into train and
        test samples
        
        df_movies: {pandas-dataframe} dataframe with movies and their genres and names
        
        df_user: {pandas-dataframe} dataframe with users and some information about them
        
        Returns
        -------
        df: {pandas-dataframe} list of dataframe with all features that we need
    """
    df = []
    for i in range(len(df_list)):
        df.append(add_my_features(df_list[i], df_users, df_movies))
    return df


def add_features(df_list, users_ind, movies_ind, df_users, df_movies):
    """Adding all features which we need by task for list of dataframe
        
        Parameters
        ----------
        df_list: {pandas-dataframe} dataframe with ratings splited into train and
        test samples
        
        users_ind: {array-like} users indexes by in splited part
        
        movies_ind: {array-like} movies indexes by in splited part
        
        df_movies: {pandas-dataframe} dataframe with movies and their genres and names
        
        df_user: {pandas-dataframe} dataframe with users and some information about them
        
        Returns
        -------
        df: {pandas-dataframe} list of dataframes with all features that we need
    """
    df = []
    for i in range(len(df_list)):
        print ('--' * 30)
        print (' ' * 15, 'Preprocessing of %d sample' % (i+1))
        print ('--' * 30)
        df.append(add_feature_for_one_df(df_list[i], users_ind[i], movies_ind[i], df_users, df_movies).drop('timestamp', axis=1))
    return df
