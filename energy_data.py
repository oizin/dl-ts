# we'll use the electricity data for this, import:
import numpy as np
import pandas as pd


def import_energy_data():
    """
    Import the UCI ML data archive Energy dataset
    Args:
        downsample: sample to one row per hour (else every 10 minutes)
    """
    
    # download
    energy_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')

    # fix data types
    energy_df['date'] = pd.to_datetime(energy_df['date'])
    energy_df['month'] = energy_df['date'].dt.month.astype(int)
    energy_df['day_of_month'] = energy_df['date'].dt.day.astype(int)

    # day_of_week=0 corresponds to Monday
    energy_df['day_of_week'] = energy_df['date'].dt.dayofweek.astype(int)
    energy_df['hour_of_day'] = energy_df['date'].dt.hour.astype(int)

    # filter columns
    selected_columns = ['date', 'day_of_week', 'hour_of_day', 'Appliances']
    energy_df = energy_df[selected_columns]
    
    # downsample to one hour
    energy_df = energy_df.set_index('date').resample('1H').mean()
    energy_df['date'] = energy_df.index
    
    # model log outcome
    energy_df['log_energy_consumption'] = np.log(energy_df['Appliances'])
    
    datetime_columns = ['date', 'day_of_week', 'hour_of_day']
    target_column = 'log_energy_consumption'
    feature_columns = datetime_columns + ['log_energy_consumption']
    energy_df = energy_df[feature_columns]
    
    return energy_df



