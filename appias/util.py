""" utility functions """
import pandas as pd


def memory_usage(df):
    """ Calculate memory used by pd.DataFrame.
    Params:
        df: pd.DataFrame
    Returns:
        str memory usage
    """
    usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    return '{:0.2f} MB'.format(usage)


def reduce_memory_usage(df, verbose=0):
    """ Reduces memory used by pd.DataFrame.
    Params:
        df: pd.DataFrame
    """
    if verbose > 0:
        print('Before Reduction:', memory_usage(df))

    df_float_cols = df.select_dtypes(include='float').columns
    df[df_float_cols] = df[df_float_cols].apply(pd.to_numeric, downcast='float')

    df_int_cols = df.select_dtypes(include='int').columns
    df[df_int_cols] = df[df_int_cols].apply(pd.to_numeric, downcast='unsigned')

    if verbose > 0:
        print('After Reduction:', memory_usage(df))

    return df
