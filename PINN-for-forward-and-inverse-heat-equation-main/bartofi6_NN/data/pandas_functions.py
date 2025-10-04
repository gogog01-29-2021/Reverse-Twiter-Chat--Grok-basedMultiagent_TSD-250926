import pandas as pd
import os


def move_columns_to_front(dataframe, list_with_column_names_to_move_to_front):
    list_with_remaining_column_names_that_are_not_moved_to_the_front = [
        column_name for column_name in dataframe.columns if column_name not in list_with_column_names_to_move_to_front
    ]

    dataframe_with_columns_that_are_not_moved_to_the_front = dataframe.loc[:, list_with_remaining_column_names_that_are_not_moved_to_the_front]
    dataframe_with_columns_that_are_moved_to_the_front = dataframe.loc[:, list_with_column_names_to_move_to_front]

    return pd.concat([dataframe_with_columns_that_are_moved_to_the_front, dataframe_with_columns_that_are_not_moved_to_the_front],
                     axis="columns")


def write_dataframe_to_file(dataframe, file_path, **kwargs_for_dataframe_to_file_function):
    file_dir_path = os.path.dirname(file_path)
    if not os.path.exists(file_dir_path):
        os.makedirs(file_dir_path)

    with open(file_path, "w") as file:
        dataframe.to_string(buf=file, **kwargs_for_dataframe_to_file_function)


def move_columns_to_front_and_write_dataframe_to_file(
    dataframe,
    file_path,
    list_with_column_names_to_move_to_front,
    **kwargs_for_dataframe_to_file_function
):
    dataframe_with_new_column_order = move_columns_to_front(dataframe, list_with_column_names_to_move_to_front)
    write_dataframe_to_file(dataframe_with_new_column_order, file_path, **kwargs_for_dataframe_to_file_function)
