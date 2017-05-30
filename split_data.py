import pandas as pd

def get_train_test_split(df, users_id, train_coeff = 0.8):
    """Spliting dataframe into two datarames with fixed size. All data sorted by time and each
        train samle contain the same portion of each user's data
        
        Parameters
        ----------
        df: {pandas-dataframe} dataframe needed to be split
        
        users_id: {list, numpy-array} list contained set of users by id
        
        train_coeff: {int} coefficient of train sample size
        
        Returns
        -------
        train, test: {pandas-dataframe} dataframes which split by time. test dataframe contains 
        elder user's ratings.
    """
    
    cur_df = df.copy()
    train = pd.DataFrame()
    test = pd.DataFrame()
    cur_df.sort_values(by='timestamp', inplace=True)
    for user in users_id:
        sub_table = cur_df[cur_df.user_id == user]
        cur_train_size = int(train_coeff * sub_table.shape[0])
        train = train.append(sub_table.head(cur_train_size))
        test = test.append(sub_table.tail(sub_table.shape[0] - cur_train_size))    
    return train, test

def get_new_indexes(df_list, ind):
    """By splited df gives an unique elemens from ind column
        
        Parameters
        ----------
        df_list: {pandas-dataframe} splited on test and train dataframe
        
        ind: {list, numpy-array} name of column
        
        Returns
        -------
        {indexes-like}: two arrays with unique element contained in ind column
        """
    return pd.Int64Index(list(set(df_list[0][ind]))), \
        pd.Int64Index(list(set(df_list[1][ind])))

def get_new_indexes_for_folds(folds, ind):
    """By splited df gives an unique elemens from ind column
        
        Parameters
        ----------
        df_list: {pandas-dataframe} splited on test and train dataframe
        
        ind: {list, numpy-array} name of column
        
        Returns
        -------
        {indexes-like}: folds arrays with unique element contained
        in ind column
    """
    indexes = []
    for i in range(len(folds)):
        indexes.append(pd.Int64Index(list(set(folds[i][ind]))))
    
    return indexes

def get_folds(df, users_id, num_folds=5):
    """Spliting dataframe into two datarames with fixed size. All data sorted by time and each
    train samle contain the same portion of each user's data
    
    Parameters
    ----------
    df: {pandas-dataframe} dataframe needed to be split
    
    users_id: {list, numpy-array} list contained set of users by id
    
    num-folds: {int} number of cross validation folds
    
    Returns
    -------
    folds: {list of dataframes} dataframes which split by time. each next fold contains later 
    user rates.
    """
    folds = []
    for i in range(num_folds):
        folds.append(pd.DataFrame())
    cur_df = df.copy()
    cur_df.sort_values(by='timestamp', inplace=True)
    for user in users_id:
        sub_table = cur_df[cur_df.user_id == user]
        indexes = sub_table.index
        size = int(sub_table.shape[0] / num_folds)
        for i in range(num_folds):
            folds[i] = folds[i].append(sub_table.loc[indexes[i*size:(i+1)*size]])
    return folds

