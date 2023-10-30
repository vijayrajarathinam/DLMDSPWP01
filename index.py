import pandas as pd
import numpy as np
import sqlite3
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource
from sqlalchemy import create_engine, MetaData, Table, Column, Float
from sklearn.linear_model import LinearRegression


# 20231026_vijay_rajarathinam_9212962_DLMDSPWP01
class DataProcessor:
    def __init__(
        self, db_file, training_data_file, ideal_functions_file, test_data_file
    ):
        self.db_file = db_file
        self.training_data_file = training_data_file
        self.ideal_functions_file = ideal_functions_file
        self.test_data_file = test_data_file
        self.chosen_ideal_functions = []

    def create_database(self):
        # Create a SQLite database and define tables for training data and ideal functions
        engine = create_engine(f"sqlite:///{self.db_file}")
        metadata = MetaData()

        # Define a table for training data (Table 1)
        training_data_table = Table(
            "training_data",
            metadata,
            Column("X", Float),
            Column("Y1", Float),
            Column("Y2", Float),
            Column("Y3", Float),
            Column("Y4", Float),
        )
        metadata.create_all(engine)

        # Load training data into the database
        self.load_data_into_db(self.training_data_file, "training_data", engine)

        # Define a table for ideal functions (Table 2)
        ideal_functions_table = Table(
            "ideal_functions",
            metadata,
            Column("X", Float),
            Column("Y1", Float),
            Column("Y2", Float),
            Column("Y3", Float),
            Column("Y4", Float),
        )
        metadata.create_all(engine)

        # Load ideal functions into the database
        self.load_data_into_db(self.ideal_functions_file, "ideal_functions", engine)

    def load_data_into_db(self, csv_file, table_name, engine):
        conn = engine.connect()
        df = pd.read_csv(csv_file)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()

    # Implement ideal function selection logic here
    # Calculate deviations and choose the best functions
    def choose_ideal_functions(self):
        # Load the training data and ideal functions into DataFrames
        training_data = pd.read_csv(self.training_data_file)
        ideal_functions = pd.read_csv(self.ideal_functions_file)

        # Initialize a list to store the results
        best_fit_functions = []

        # Loop through the ideal functions and perform linear regression
        for function_name in ideal_functions.columns[1:5]:
            # Skip the first column which contains X values
            # Extract X and Y values for the current ideal function
            x = training_data["x"].values.reshape(-1, 1)
            y = training_data[function_name].values

            # Fit a linear regression model
            model = LinearRegression()
            model.fit(x, y)

            # Calculate the sum of squared deviations
            deviations = y - model.predict(x)
            sum_squared_deviations = (deviations**2).sum()

            # Store the results
            best_fit_functions.append(
                (
                    function_name,
                    model.coef_[0],
                    model.intercept_,
                    sum_squared_deviations,
                )
            )

        # Sort the functions by the sum of squared deviations and choose the top four
        best_fit_functions.sort(key=lambda x: x[3])
        chosen_functions = best_fit_functions[:4]

        # Store the chosen ideal functions
        for function_name, slope, intercept, sum_squared_deviations in chosen_functions:
            self.chosen_ideal_functions.append(
                [function_name, slope, intercept, sum_squared_deviations]
            )

    # Implement mapping of test data to chosen ideal functions
    # Calculate deviations and store the results in the database
    def map_test_data(self):
        # Load the test data into a DataFrame
        test_data = pd.read_csv(self.test_data_file)

        # Initialize a list to store the mapping and deviation results
        results = []

        # Loop through each x-y pair in the test data
        for _, row in test_data.iterrows():
            x = row["x"]
            y = row["y"]

            # Initialize variables to track the best match and its deviation
            best_match = None
            best_deviation = None

            # Calculate deviations for each of the four chosen ideal functions
            for function_name, slope, intercept, _ in self.chosen_ideal_functions:
                predicted_y = slope * x + intercept
                deviation = abs(y - predicted_y)

                if best_deviation is None or deviation < best_deviation:
                    best_deviation = deviation
                    best_match = function_name

            # Check if the best deviation is within the acceptable range
            largest_training_deviation = max(
                self.chosen_ideal_functions, key=lambda x: x[3]
            )[3]
            acceptable_range = np.sqrt(2) * largest_training_deviation

            if best_deviation <= acceptable_range:
                results.append([x, y, best_match, best_deviation])

        # Create a DataFrame from the results
        result_df = pd.DataFrame(
            results, columns=["X", "Y", "Assigned_Function", "Deviation"]
        )

        # Save the results to a CSV file
        result_df.to_csv("mapping_results.csv", index=False)

    def visualize_data(self):
        # Load the mapping results (assuming you've already calculated and saved them)
        mapping_results = pd.read_csv("mapping_results.csv")

        # Create a Bokeh figure
        p = figure(
            title="Mapping Test Data to Ideal Functions",
            x_axis_label="X",
            y_axis_label="Y",
        )

        # Create a color mapping for different ideal functions
        color_mapper = {
            function_name: function_name
            for function_name, _, _, _ in self.chosen_ideal_functions
        }

        # Create a data source for the scatter plot
        source = ColumnDataSource(
            data=dict(
                X=mapping_results["X"],
                Y=mapping_results["Y"],
                Assigned_Function=mapping_results["Assigned_Function"],
                Deviation=mapping_results["Deviation"],
                Color=[
                    color_mapper[function_name]
                    for function_name in mapping_results["Assigned_Function"]
                ],
            )
        )

        # Create a scatter plot for mapping results
        p.scatter(
            x="X",
            y="Y",
            source=source,
            size=8,
            color="Color",
            legend_field="Assigned_Function",
        )

        # Add a legend to the plot
        p.legend.title = "Assigned Function"
        p.legend.label_text_font_size = "10pt"

        # Customize the plot appearance
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"

        # Save the plot to an HTML file (optional)
        output_file("mapping_results_plot.html")

    def run(self):
        self.create_database()
        self.choose_ideal_functions()
        self.map_test_data()
        self.visualize_data()

    # Save the mapping and deviation results to a table in the SQLite database
    def save_results(self, result_file):
        engine = create_engine(f"sqlite:///{self.db_file}")
        metadata = MetaData()

        # Define a table for result data (Table 1)
        Table(
            "mapping_results",
            metadata,
            Column("X", Float),
            Column("Y1", Float),
            Column("Y2", Float),
            Column("Y3", Float),
            Column("Y4", Float),
        )
        metadata.create_all(engine)

        # Load training data into the database
        self.load_data_into_db("mapping_results.csv", "mapping_results", engine)


class ResultDatabase(DataProcessor):
    # Overridden method to save results in a database
    # You can implement the logic to save results in a separate table
    def save_results(self, result_file):
        # Create a DataFrame from the results
        result_df = pd.DataFrame(
            pd.read_csv("mapping_results.csv"),
            columns=["X", "Y", "Assigned_Function", "Deviation"],
        )

        # Save the results to a CSV file
        result_df.to_csv(result_file, index=False)


if __name__ == "__main__":
    db_file = "database.db"
    training_data_file = "train.csv"
    ideal_functions_file = "ideal.csv"
    test_data_file = "test.csv"

    processor = ResultDatabase(
        db_file, training_data_file, ideal_functions_file, test_data_file
    )
    processor.run()
    processor.save_results("results.csv")
