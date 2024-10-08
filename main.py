

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, select, MetaData, Table


class DataProcessorBase:
    """
    A base class for processing data.

    Attributes:
        train_data (pd.DataFrame): Training data.
        ideal_data (pd.DataFrame): Ideal data.
        test_data (pd.DataFrame): Test data.
    """

    def __init__(self, train_path, ideal_path, test_path, database_url):
        """
        Initializes the DataProcessorBase object.

        Args:
            train_path (str): Table name for the training data in the database.
            ideal_path (str): Table name for the ideal data in the database.
            test_path (str): Table name for the test data in the database.
            database_url (str): SQLAlchemy-compatible database URL.
        """
        self.train_data = self.load_data_from_sql(train_path, database_url)
        self.ideal_data = self.load_data_from_sql(ideal_path, database_url)
        self.test_data = self.load_data_from_sql(test_path, database_url, True)

    def load_data_from_sql(self, table_name, database_url, skip_index=False):
        """
        Loads data from an SQL table using SQLAlchemy.

        Args:
            table_name (str): Name of the table in the database.
            database_url (str): SQLAlchemy-compatible database URL.
            skip_index (bool): Whether to skip the index column.

        Returns:
            pd.DataFrame: Loaded data.
        """
        database_url = "sqlite:///database.db"

        # Replace 'your_table_name' with the actual name of your table
        table_name = table_name

        # Create an SQLAlchemy engine
        engine = create_engine(database_url)
        
        with engine.connect() as conn, conn.begin():  
            data = pd.read_sql_table(table_name, conn)

        return data

    def preprocess_data(self, data):
        """
        Preprocesses the data by removing NaN values and normalizing it.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The preprocessed and normalized data.
        """
        data = data.dropna()
        normalized_data = (data - data.mean()) / data.std()
        return normalized_data

    def calculate_squared_errors(self, column1, column2):
        """
        Calculates the sum of squared errors between two columns.

        Args:
            column1 (pd.Series): First column.
            column2 (pd.Series): Second column.

        Returns:
            float: Sum of squared errors.
        """
        if len(column1) != len(column2):
            raise ValueError("Columns must have the same length.")
        squared_errors = (np.array(column1) - np.array(column2))**2
        sum_squared_errors = np.sum(squared_errors)
        return sum_squared_errors

    def find_best_fit_column(self, normalized_train, normalized_ideal):
        """
        Finds the best-fit column index for each column in the training data.

        Args:
            normalized_train (pd.DataFrame): Normalized training data.
            normalized_ideal (pd.DataFrame): Normalized ideal data.

        Returns:
            np.array: Array of best-fit column indices.
        """
        best_fit_indices = []
        for train_col_index in range(1, len(normalized_train.columns)):
            train_column = normalized_train.iloc[:, train_col_index]
            sum_squared_errors_list = []

            for ideal_col_index in range(1, len(normalized_ideal.columns)):
                ideal_column = normalized_ideal.iloc[:, ideal_col_index]
                sum_squared_errors = self.calculate_squared_errors(train_column, ideal_column)
                sum_squared_errors_list.append(round(sum_squared_errors, 5))

            print("Train Column Index ", train_col_index)
            best_fit_index = np.argmin(sum_squared_errors_list)
            print("Best Fit Index Position from the Ideal Data: ", best_fit_index+1)
            best_fit_indices.append(best_fit_index)
            print("Minimum Sum of Squared Errors: ", min(sum_squared_errors_list))
            print("------------------------------------")

        return np.array(best_fit_indices) + 1

    def plot_data(self, x_axis_range, normalized_ideal, normalized_train, best_fit_indices):
        """
        Plots the data for visualization.

        Args:
            x_axis_range (range): Range for x-axis.
            normalized_ideal (pd.DataFrame): Normalized ideal data.
            normalized_train (pd.DataFrame): Normalized training data.
            best_fit_indices (np.array): Array of best-fit column indices.
        """
        for ideal_col_index in range(4):
            plt.title('Y'+str( best_fit_indices[ideal_col_index]))
            plt.plot(x_axis_range, normalized_ideal.iloc[:, best_fit_indices[ideal_col_index]])
            plt.plot(x_axis_range, normalized_train.iloc[:, ideal_col_index + 1])
            plt.show()

    def calculate_min_distance(self, array_a, value_b):
        """
        Calculates the minimum absolute distance between an array and a value.

        Args:
            array_a (np.array): Array to compare.
            value_b (float): Value to compare against.

        Returns:
            float: Minimum absolute distance.
        """
        distances = [np.abs(array_a[i] - value_b) for i in range(len(array_a))]
        min_distance = np.min(distances)
        return min_distance


class DerivedDataProcessor(DataProcessorBase):
    """
    A derived class that extends DataProcessorBase with additional functionality.
    """

    def __init__(self, train_path, ideal_path, test_path, database_url):
        """
        Initializes the DerivedDataProcessor object.

        Args:
            train_path (str): Table name for the training data in the database.
            ideal_path (str): Table name for the ideal data in the database.
            test_path (str): Table name for the test data in the database.
            database_url (str): SQLAlchemy-compatible database URL.
        """
        super().__init__(train_path, ideal_path, test_path, database_url)
        self.normalized_train_data = None
        self.normalized_ideal_data = None
        self.normalized_test_data = None
        self.best_fit_indices = None

    def classify_test_data(self, ideal_data_columns):
        """
        Classifies the test data based on the ideal data.

        Args:
            ideal_data_columns (pd.DataFrame): DataFrame containing the ideal data columns.

        Returns:
            list: List of classified labels for the test data with deviation.
        """
        distances = []
        indices = []

        for test_row_index in range(len(self.normalized_test_data)):
            min_distances = []

            for ideal_col_index in range(4):
                min_distances.append(
                    self.calculate_min_distance(ideal_data_columns.iloc[:, ideal_col_index],
                                                self.normalized_test_data.loc[test_row_index]))

            distances.append(round(min(min_distances), 3))
            indices.append(np.argmin(min_distances))

        classified_labels = ['Y' + str(self.best_fit_indices[index]) for index in indices]
        return classified_labels,distances
        

    def main(self):
        """
        The main workflow method that performs data processing and classification.
        """
        self.normalized_train_data = self.preprocess_data(self.train_data)
        self.normalized_ideal_data = self.preprocess_data(self.ideal_data)
        self.normalized_test_data = self.preprocess_data(self.test_data)

        self.best_fit_indices = self.find_best_fit_column(self.normalized_train_data, self.normalized_ideal_data)

        x_axis_range = range(0, 400)

        self.plot_data(x_axis_range, self.normalized_ideal_data, self.normalized_train_data, self.best_fit_indices)

        classified_labels,deviation = self.classify_test_data(self.normalized_ideal_data.iloc[:, self.best_fit_indices])
        # print(classified_labels)

        final_test_data = self.test_data.copy()
        final_test_data["No. of ideal func"] = classified_labels
        final_test_data["Delta Y (test func)"] = deviation
        return final_test_data


if __name__ == "__main__":
    database_url = "sqlite:///database.db"
    # Replace the placeholder values with your actual database URL and table names
    processor = DerivedDataProcessor("train_table", "ideal_table", "test_table", database_url)
    final_result = processor.main()
    print(final_result)

    #Store result into DB file
    engine = create_engine(database_url)
    # Create a metadata instance
    metadata = MetaData()
    table_name = "result"
    engine = create_engine(database_url)
    final_result.to_sql(table_name, engine, index=False, if_exists="replace")
    print("final result loaded in database")
