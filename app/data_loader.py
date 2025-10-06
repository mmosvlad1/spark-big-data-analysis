from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
)
import os


TRIP_SCHEMA: StructType = StructType(
    [
        StructField("medallion", StringType(), True),
        StructField("hack_license", StringType(), True),
        StructField("vendor_id", StringType(), True),
        StructField("rate_code", StringType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("pickup_datetime", TimestampType(), True),
        StructField("dropoff_datetime", TimestampType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("trip_time_in_secs", IntegerType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
    ]
)


FARE_SCHEMA: StructType = StructType(
    [
        StructField("medallion", StringType(), True),
        StructField("hack_license", StringType(), True),
        StructField("vendor_id", StringType(), True),
        StructField("pickup_datetime", TimestampType(), True),
        StructField("payment_type", StringType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("surcharge", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
    ]
)


def _resolve_path(base_dir: str, relative: str) -> str:
    if os.path.isabs(relative):
        return relative
    return os.path.join(base_dir, relative)


def _pick_one_file(directory: str, prefix: str, exact_filename: str) -> str:
    exact_path = os.path.join(directory, exact_filename)
    if os.path.isfile(exact_path):
        return exact_path
    for name in sorted(os.listdir(directory)):
        if name.startswith(prefix) and name.endswith(".csv"):
            return os.path.join(directory, name)
    raise FileNotFoundError(f"No CSV found in {directory} with prefix {prefix}")


def load_trip_data(
    spark: SparkSession, data_root: str = "/data/nyc", test: bool = False
) -> DataFrame:
    trip_dir = _resolve_path(data_root, "tripData2013")
    if test:
        path = _pick_one_file(
            trip_dir, prefix="trip_data_", exact_filename="trip_data_1.csv"
        )
    else:
        path = os.path.join(trip_dir, "trip_data_*.csv")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .schema(TRIP_SCHEMA)
        .csv(path)
    )
    return df


def load_fare_data(
    spark: SparkSession, data_root: str = "/data/nyc", test: bool = False
) -> DataFrame:
    fare_dir = _resolve_path(data_root, "faredata2013")
    if test:
        path = _pick_one_file(
            fare_dir, prefix="trip_fare_", exact_filename="trip_fare_1.csv"
        )
    else:
        path = os.path.join(fare_dir, "trip_fare_*.csv")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .schema(FARE_SCHEMA)
        .csv(path)
    )
    return df


def validate_dataframes(trip_df: DataFrame, fare_df: DataFrame) -> None:
    trip_df.printSchema()
    fare_df.printSchema()
    trip_df.show(5, truncate=False)
    fare_df.show(5, truncate=False)
