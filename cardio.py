from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Cardiovascular Disease Analysis") \
    .getOrCreate()

# Load the dataset
df = spark.read.csv("/opt/spark/cardio_train.csv", header=True, sep=';')

# Show schema and data preview
df.printSchema()
df.show(5)

# Step 1: Handling Missing Values

# Get total rows in the dataset
total_rows = df.count()

# Count the number of missing values per column
missing_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])

# Calculate the percentage of missing values per column
missing_percentage = missing_counts.select([(col(c) / total_rows * 100).alias(c) for c in missing_counts.columns])

# Show the percentage of missing values for each column
missing_percentage.show()

# Set thresholds for dropping and imputing
drop_threshold = 30  # Drop columns if missing values > 30%
impute_threshold = 5  # Impute columns if missing values <= 5%

# Drop or impute columns based on the missing percentage
for column in df.columns:
    missing_value_percentage = missing_percentage.select(col(column)).collect()[0][0]

    if missing_value_percentage > drop_threshold:
        print(f"Dropping column {column} (Missing {missing_value_percentage:.2f}%)")
        df = df.drop(column)

    elif missing_value_percentage <= impute_threshold:
        print(f"Imputing column {column} (Missing {missing_value_percentage:.2f}%) with mean")
        mean_value = df.select(avg(col(column))).collect()[0][0]
        df = df.fillna({column: mean_value})

# Step 2: Normalization of Critical Features (Cholesterol and Blood Pressure)

from pyspark.ml.feature import StandardScaler, VectorAssembler

# Convert relevant columns to numeric types for scaling
df = df.withColumn("cholesterol", col("cholesterol").cast("float")) \
       .withColumn("ap_hi", col("ap_hi").cast("float")) \
       .withColumn("ap_lo", col("ap_lo").cast("float"))

# Assemble the features to be normalized into a single vector column
assembler = VectorAssembler(inputCols=["cholesterol", "ap_hi", "ap_lo"], outputCol="features")

# Transform the DataFrame with assembled features
df_assembled = assembler.transform(df)

# Apply StandardScaler for normalization
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# Show scaled features
df_scaled.select("scaled_features").show(5)

# Step 3: Statistical Analysis (Correlation)

# Convert columns to numeric types for correlation analysis
df_cleaned = df_scaled.withColumn("age", col("age").cast("float")) \
                      .withColumn("cardio", col("cardio").cast("int"))

# Calculate correlations between relevant features and 'cardio'
correlations = {}
for col1 in ['age', 'cholesterol', 'ap_hi', 'ap_lo']:
    corr_value = df_cleaned.stat.corr(col1, 'cardio')
    correlations[col1] = corr_value
    print(f"Correlation between {col1} and cardio: {corr_value:.4f}")

# Step 4: Visualize Correlations using a Heatmap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create Pandas DataFrame with correlations for heatmap
corr_data = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Cardio'])

# Plot heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', cbar=True)
plt.title("Correlations with Cardiovascular Disease")
plt.show()


df.write.csv("/opt/spark/processedData.csv", header=True)
