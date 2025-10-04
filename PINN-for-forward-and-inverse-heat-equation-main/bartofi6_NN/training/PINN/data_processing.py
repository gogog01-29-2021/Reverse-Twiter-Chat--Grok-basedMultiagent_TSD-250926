import pandas as pd
import os
from ...data.pandas_functions import move_columns_to_front_and_write_dataframe_to_file


def sort_results_from_main_csv_file(
    list_with_column_names_for_individual_sorting,
    list_with_column_names_to_put_after_the_first_sorted_column,
    ascending: bool or list(bool) = True
):
    dataframe_from_main_csv = pd.read_csv(os.path.join(".", "main.csv"), header=0, index_col=False)

    for column_index, name_of_column_to_sort in enumerate(list_with_column_names_for_individual_sorting):
        sort_in_ascending_order_for_current_column = ascending[column_index] if isinstance(ascending, list) else ascending

        sorted_dataframe = dataframe_from_main_csv.sort_values(name_of_column_to_sort, axis="rows",
                                                               ascending=sort_in_ascending_order_for_current_column)

        move_columns_to_front_and_write_dataframe_to_file(
            sorted_dataframe,
            os.path.join(".", f"sorted by {name_of_column_to_sort}.txt"),
            [name_of_column_to_sort] + list_with_column_names_to_put_after_the_first_sorted_column,
            index=False)
