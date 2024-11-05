# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# Start Spark session
spark = SparkSession.builder.appName("CardiovascularRiskPrediction_DNN").getOrCreate()

# Load data into Spark DataFrame
data = spark.read.csv("/opt/spark/cardio_train.csv", header=True, sep=";", inferSchema=True)
print("Initial Data Schema:")
data.printSchema()

# Stratified Sampling
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
train_data = train_data.sampleBy("cardio", fractions={0: 0.8, 1: 0.8}, seed=42)
test_data = test_data.sampleBy("cardio", fractions={0: 0.2, 1: 0.2}, seed=42)

# Feature selection
feature_columns = [col for col in data.columns if col not in ["id", "cardio"]]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Define DNN layer structure
dnn_layers = [len(feature_columns), 64, 32, 16, 2]
# Set up the DNN model
dnn_model = MultilayerPerceptronClassifier(featuresCol="scaledFeatures", labelCol="cardio",
                                           layers=dnn_layers, blockSize=128, maxIter=100)

# Evaluation setup
binary_evaluator = BinaryClassificationEvaluator(labelCol="cardio")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="cardio", metricName="accuracy")

# Pipeline setup
pipeline = Pipeline(stages=[assembler, scaler, dnn_model])

# Fit the model on training data
fitted_dnn_model = pipeline.fit(train_data)

# Predictions on test data
predictions = fitted_dnn_model.transform(test_data)
predictions.select("scaledFeatures", "cardio", "prediction").show(5)

# Model evaluation
accuracy = multi_evaluator.evaluate(predictions)
auc = binary_evaluator.evaluate(predictions)

# Calculate precision and recall
tp = predictions.filter((col("cardio") == 1) & (col("prediction") == 1)).count()
fp = predictions.filter((col("cardio") == 0) & (col("prediction") == 1)).count()
fn = predictions.filter((col("cardio") == 1) & (col("prediction") == 0)).count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

print(f"DNN Model - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
print(f"DNN Model - Precision: {precision:.4f}, Recall: {recall:.4f}")

# Visualize performance metrics
metrics_df = pd.DataFrame([["DNN", accuracy, auc, precision, recall]],
                          columns=["Model", "Accuracy", "AUC", "Precision", "Recall"])
metrics_df.plot(x='Model', kind='bar', figsize=(8, 5), title='DNN Model Performance Metrics',
                y=["Accuracy", "Precision", "Recall"])
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Plot ROC Curve for DNN
if "probability" in predictions.columns:
    y_true = predictions.select('cardio').toPandas()
    y_scores = predictions.select('probability').rdd.map(lambda x: float(x['probability'][1])).collect()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'DNN (AUC = {roc_auc_score(y_true, y_scores):.2f})')
    plt.title('ROC Curve for DNN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

# Stop Spark session
spark.stop()
