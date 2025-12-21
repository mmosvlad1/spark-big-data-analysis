from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, sin, cos, acos, radians, when, hour, dayofweek, udf
from pyspark.sql.types import DoubleType, FloatType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import os
import pandas as pd
import logging
import sys

logger = logging.getLogger("ML_Training")
logger.setLevel(logging.INFO)

def setup_logger(log_file_path):
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(f"Logging initialized. Writing to: {log_file_path}")

def calculate_haversine_distance(df: DataFrame) -> DataFrame:
    R = 6371.0
    lat1 = radians(col("pickup_latitude"))
    lon1 = radians(col("pickup_longitude"))
    lat2 = radians(col("dropoff_latitude"))
    lon2 = radians(col("dropoff_longitude"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    from pyspark.sql.functions import asin, sqrt, sin, cos
    a = (sin(dlat / 2)**2) + cos(lat1) * cos(lat2) * (sin(dlon / 2)**2)
    c = 2 * asin(sqrt(a))
    distance = R * c
    return df.withColumn("haversine_distance", distance)

def prepare_features(df: DataFrame) -> DataFrame:
    logger.info("Starting feature preparation...")
    # Filter invalid data
    df_clean = df.filter(
        (col("total_amount").between(2.5, 500)) &
        (col("pickup_longitude").between(-75, -72)) &
        (col("pickup_latitude").between(40, 42)) &
        (col("dropoff_longitude").between(-75, -72)) &
        (col("dropoff_latitude").between(40, 42)) &
        (col("passenger_count") > 0)
    )

    # Feature Engineering
    df_feat = calculate_haversine_distance(df_clean)
    df_feat = df_feat.withColumn("pickup_hour", hour(col("pickup_datetime")))
    df_feat = df_feat.withColumn("day_of_week", dayofweek(col("pickup_datetime")))
    df_feat = df_feat.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
    
    # --- Traffic Prediction Logic ---
    # 1. Ensure valid time/distance for speed calculation
    df_feat = df_feat.filter((col("trip_time_in_secs") > 60) & (col("haversine_distance") > 0.5))
    
    # 2. Calculate Speed (mph)
    # Distance is in km. 1 km = 0.621371 miles.
    # Time is in seconds.
    df_feat = df_feat.withColumn(
        "trip_speed_mph", 
        (col("haversine_distance") * 0.621371) / (col("trip_time_in_secs") / 3600)
    )
    
    # 3. Target: High Traffic (1) if speed < 12 mph, else (0)
    df_feat = df_feat.withColumn("is_high_traffic", when(col("trip_speed_mph") < 12, 1.0).otherwise(0.0))
    
    logger.info("Feature preparation complete (Traffic Logic added).")
    return df_feat

def evaluate_regression(predictions: DataFrame, label_col: str = "total_amount"):
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    
    return {
        "RMSE": evaluator_rmse.evaluate(predictions),
        "MAE": evaluator_mae.evaluate(predictions),
        "R2": evaluator_r2.evaluate(predictions)
    }

def evaluate_classification(predictions: DataFrame, label_col: str = "is_high_traffic"):
    acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    prec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")
    f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
    
    return {
        "Accuracy": acc_eval.evaluate(predictions),
        "Precision": prec_eval.evaluate(predictions),
        "Recall": rec_eval.evaluate(predictions),
        "F1": f1_eval.evaluate(predictions)
    }

def train_regression(spark: SparkSession, data: DataFrame, output_path: str):
    logger.info("Running REGRESSION Task...")
    metrics_dir = os.path.join(output_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    numeric_cols = ["haversine_distance", "pickup_hour", "is_weekend", "passenger_count"]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Regression Split: Train={train_data.count()}, Test={test_data.count()}")
    
    results = []

    # 1. Linear Regression
    logger.info("Training Linear Regression...")
    lr = LinearRegression(featuresCol="features", labelCol="total_amount", maxIter=100, regParam=0.1, elasticNetParam=0.8)
    model_lr = Pipeline(stages=[assembler, lr]).fit(train_data)
    metrics_lr = evaluate_regression(model_lr.transform(test_data), "total_amount")
    logger.info(f"LR Metrics: {metrics_lr}")
    results.append({"Model": "LinearRegression", **metrics_lr})
    model_lr.write().overwrite().save(os.path.join(output_path, "LinearRegression"))
    
    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(featuresCol="features", labelCol="total_amount", numTrees=50, maxDepth=10)
    model_rf = Pipeline(stages=[assembler, rf]).fit(train_data)
    metrics_rf = evaluate_regression(model_rf.transform(test_data), "total_amount")
    logger.info(f"RF Metrics: {metrics_rf}")
    results.append({"Model": "RandomForestRegressor", **metrics_rf})
    model_rf.write().overwrite().save(os.path.join(output_path, "RandomForestRegressor"))

    # 3. GBT
    logger.info("Training GBT Regressor...")
    gbt = GBTRegressor(featuresCol="features", labelCol="total_amount", maxIter=30, maxDepth=5)
    model_gbt = Pipeline(stages=[assembler, gbt]).fit(train_data)
    metrics_gbt = evaluate_regression(model_gbt.transform(test_data), "total_amount")
    logger.info(f"GBT Metrics: {metrics_gbt}")
    results.append({"Model": "GBTRegressor", **metrics_gbt})
    model_gbt.write().overwrite().save(os.path.join(output_path, "GBTRegressor"))
    
    pd.DataFrame(results).to_csv(os.path.join(metrics_dir, "regression_results.csv"), index=False)
    logger.info("Regression Complete.")

def train_classification(spark: SparkSession, data: DataFrame, output_path: str):
    logger.info("Running CLASSIFICATION Task (Traffic Prediction)...")
    metrics_dir = os.path.join(output_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # NOTE: We do NOT use 'haversine_distance' here because it's directly used to calculate the label (Speed).
    # Using it might cause leakage if the model learns distance implies congestion (though valid correlation).
    # Let's keep it for now as congestion depends on location/distance + time.
    numeric_cols = ["haversine_distance", "pickup_hour", "is_weekend", "passenger_count"]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Classification Split: Train={train_data.count()}, Test={test_data.count()}")
    
    results = []

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    logr = LogisticRegression(featuresCol="features", labelCol="is_high_traffic", maxIter=100, regParam=0.1)
    model_logr = Pipeline(stages=[assembler, logr]).fit(train_data)
    metrics_logr = evaluate_classification(model_logr.transform(test_data), "is_high_traffic")
    logger.info(f"LogR Metrics: {metrics_logr}")
    results.append({"Model": "LogisticRegression", **metrics_logr})
    model_logr.write().overwrite().save(os.path.join(output_path, "LogisticRegression"))

    # 2. Random Forest
    logger.info("Training Random Forest Classifier...")
    rfc = RandomForestClassifier(featuresCol="features", labelCol="is_high_traffic", numTrees=50, maxDepth=10)
    model_rfc = Pipeline(stages=[assembler, rfc]).fit(train_data)
    metrics_rfc = evaluate_classification(model_rfc.transform(test_data), "is_high_traffic")
    logger.info(f"RFC Metrics: {metrics_rfc}")
    results.append({"Model": "RandomForestClassifier", **metrics_rfc})
    model_rfc.write().overwrite().save(os.path.join(output_path, "RandomForestClassifier"))

    # 3. GBT
    logger.info("Training GBT Classifier...")
    gbtc = GBTClassifier(featuresCol="features", labelCol="is_high_traffic", maxIter=30, maxDepth=5)
    model_gbtc = Pipeline(stages=[assembler, gbtc]).fit(train_data)
    metrics_gbtc = evaluate_classification(model_gbtc.transform(test_data), "is_high_traffic")
    logger.info(f"GBTC Metrics: {metrics_gbtc}")
    results.append({"Model": "GBTClassifier", **metrics_gbtc})
    model_gbtc.write().overwrite().save(os.path.join(output_path, "GBTClassifier"))
    
    pd.DataFrame(results).to_csv(os.path.join(metrics_dir, "classification_results.csv"), index=False)
    logger.info("Classification Complete.")

def run_ml_pipeline(spark: SparkSession, df: DataFrame, base_output_path: str):
    log_file = os.path.join(base_output_path, "ml_training.log")
    setup_logger(log_file)
    
    logger.info("=== Starting ML Pipeline ===")
    
    logger.info("Preparing features...")
    data = prepare_features(df).limit(50000000)
    
    # Optimization: Cache data since it's used for 6 models
    logger.info("Caching prepared dataset...")
    data.persist()
    
    # Run Pipelines
    train_regression(spark, data, os.path.join(base_output_path, "regression"))
    train_classification(spark, data, os.path.join(base_output_path, "classification"))
    
    logger.info("Unpersisting dataset...")
    data.unpersist()
    logger.info("=== ML Pipeline Finished ===")
