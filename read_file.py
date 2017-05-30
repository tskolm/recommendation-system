import pandas as pd
import zipfile

def reset_index(df):
    #use index_id
    """Rename index cells by index_id column which is contained in dataframe.
        And drop this column from dataframe.
        
        Parameters
        ----------
        df: {pandas-dataframe} dataframe with index_id column
        
        Returns
        -------
        cur_df: {pandas-dataframe} dataframe which index like index_id and this dataframe
        doesn't contain index_id column
    """
    cur_df = df
    cur_df.index = cur_df.index_id
    cur_df.index = cur_df.index.astype(int)
    cur_df = cur_df.drop('index_id', axis=1)
    return cur_df

def read_file(parent_name, parent_file, file_name, columns, encoding='utf-8', index=False):
    """Reading file from path = parent_name[:-4] + file_name. We need to throw out 
        file extension
        
        Parameters
        ----------
        parent_name: {string-like} name of archive and path to it
        
        parent_file: {string-like} python file descriptor to zipped file
        
        file_name: {string-like} file name in ./data folder which has .bat extension
        
        columns: {python-list, numpy-array} name of columns which can be found in
        data description
        
        encoding: {string-like} should be using when it's different from utf-8
        
        Returns
        -------
        df: {pandas-dataframe} which was read from parent file with name file_name
    """
    
    with parent_file.open('%s/%s.dat' % (parent_name[:-4], file_name), 'r') as f:
        read_data = [l.decode(encoding).rstrip().split('::') for l in f]
        df = pd.DataFrame(read_data, columns=columns)
    if index:
        df = reset_index(df)
    return df

def read_zipped_file(path):
    """Reading archive from path
        
        Parameters
        ----------
        path: {string-like} path to archive
        
        Returns
        -------
        df_users, df_movies, df_rates: {pandas-dataframes} dataframes are contained in archive
    """
    
    with zipfile.ZipFile(path, 'r') as zip_file:
        df_users = read_file(path, zip_file, 'users', ['index_id', 'gender', 'age', 'occupation', \
                                                       'zip_code'], index=True)
        df_movies = read_file(path, zip_file, 'movies', ['index_id', 'title', 'genres'], \
                              encoding= 'latin_1', index=True)
        df_rates = read_file(path, zip_file, 'ratings', ['user_id', 'movie_id', 'rating', 'timestamp'])
        df_rates = df_rates.astype(int)
    return df_users, df_movies, df_rates
