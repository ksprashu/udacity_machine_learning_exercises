#!/usr/bin/python

import numpy as np
import pandas as pd

def get_entropy(df):
    """ Return entropy of mobugs and lobugs in dataframe
    """

    mobug_count = df[df['Species'] == 'Mobug'].shape[0]
    lobug_count = df[df['Species'] == 'Lobug'].shape[0]
    total_count = df.shape[0]

    mobug_ratio = mobug_count / total_count
    lobug_ratio = lobug_count / total_count
    entropy = (-1 * mobug_ratio * np.log2(mobug_ratio)) \
        + (-1 * lobug_ratio * np.log2(lobug_ratio))

    return entropy

def get_infogain(df, df_child1):
    """ Calculate entropy of each child and return their weighted average
    """

    entropy_parent = get_entropy(df)

    entropy_child1 = get_entropy(df_child1)
    count_child1 = df_child1.shape[0]

    df_child2 = df[~df.isin(df_child1)].dropna()
    entropy_child2 = get_entropy(df_child2)
    count_child2 = df_child2.shape[0]

    total = count_child1 + count_child2

    weighted_entropy = ((count_child1 / total) * entropy_child1) \
        + ((count_child2 / total) * entropy_child2)

    return entropy_parent - weighted_entropy


if __name__ == "__main__":
    df = pd.read_csv('ml-bugs.csv')

    print(df.head() + '\n')
    print(df['Species'].value_counts())
    print(df['Color'].value_counts())

    print('\nentropy parent = {}'.format(np.round(get_entropy(df), 4)))

    df_blue = df[df['Color'] == 'Blue']
    print('entropy blue = ', np.round(get_infogain(df, df_blue), 4))

    df_green = df[df['Color'] == 'Green']
    print('entropy green = ', np.round(get_infogain(df, df_green), 4))

    df_brown = df[df['Color'] == 'Brown']
    print('entropy brown = ', np.round(get_infogain(df, df_brown), 4))

    df_length_lt17 = df[df['Length (mm)'] < 17.0]
    print('entropy <17 = ', np.round(get_infogain(df, df_length_lt17), 4))

    df_length_lt20 = df[df['Length (mm)'] < 20.0]
    print('entropy <20 = ', np.round(get_infogain(df, df_length_lt20), 4))

