import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class exploratory_data_analysis:
    def __init__(self, filepath) -> None:
        # Read the data
        self.data = pd.read_csv(filepath, delimiter='|')

    def _info(self):
        print(f"info of the dataset {self.data.info()}")
        row, col = self.data.shape
        print(f"The data has {col} columns and {row} rows.")
        print(f"Descriptions of the dataset: {self.data.describe()}")

    def _info_about_specified_colmn(self, *args):
        for ar in args:
            print(f"Here is the description of the {ar} column: {self.data[ar].describe()}")

    def missing_values(self):
        print(f"The missing value count for each column:\n{self.data.isnull().sum()}")
        print(f"Total missing values in dataset: {self.data.isnull().sum().sum()}")

    def handle_missing_values(self):
        numeric_col = self.data.select_dtypes(include=['number']).columns
        non_numeric_col = self.data.select_dtypes(exclude=['number']).columns
        
        for col in numeric_col:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        for col in non_numeric_col:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        return self.data.isnull().sum()

    def Univariate_analysis(self, column):
        sns.histplot(self.data[column], bins=30)
        plt.title(f'Univariate Analysis of {column}')
        plt.show()

    def Bivariate_analysis(self, *args):
        sns.scatterplot(x=args[0], y=args[1], data=self.data)
        plt.title(f'Bivariate Analysis of {args[0]} and {args[1]}')
        plt.show()

    def Multivarite_analysis(self, *args):
        sns.pairplot(self.data[[ar for ar in args]])
        plt.show()

    def detecting_outliers(self, column):
        sns.boxplot(x=column, data=self.data)
        plt.title(f'Boxplot to Detect Outliers in {column}')
        plt.show()

    def _basic_columns_ploting(self,*args):
        for ar in args:
            plt.figure(figsize=(10,6))
            sns.histplot(self.data[ar], bins=30, kde=True)
            plt.title(f"Distribution of {ar}")
            plt.show()

    def two_columns_scatter_plot(self, *args):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=args[0], y=args[1], data=self.data)
        plt.title(f"{args[0]} X {args[1]}")
        plt.show()

    def handle_outliers(self, column):
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
        return "success", self.data

    def show_outliers(self, column):
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        print(f"Number of outliers in {column}: {outliers.shape[0]}")
        print(f"Outlier values in {column}:\n{outliers[[column]]}")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data[column])
        plt.title(f"Boxplot of {column} (Outliers Highlighted)")
        plt.show()

    def loss_ratio_by_category(self, category):
        self.data['LossRatio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        result = self.data.groupby(category)['LossRatio'].mean().sort_values(ascending=False)
        print(result)
        result.plot(kind='bar', figsize=(10, 6), title=f'Average Loss Ratio by {category}')
        plt.ylabel('Loss Ratio')
        plt.show()

    def time_series_trend_plot(self, date_col, value_col):
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        trend = self.data.groupby(self.data[date_col].dt.to_period('M'))[value_col].sum()
        trend.plot(kind='line', marker='o', figsize=(12, 6), title=f'{value_col} Trend Over Time')
        plt.xlabel('Month')
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        corr = self.data.select_dtypes(include='number').corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
