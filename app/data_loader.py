from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import to_date, col, to_timestamp, date_trunc
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
    DateType,
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


def load_processed_trips(spark: SparkSession, processed_root: str = "/data/processed") -> DataFrame:
    """
    Завантажує вже оброблені поїздки з Parquet:
    очікується структура processed_root/nyc_taxi_trips/partitioned(year,month,day)
    """
    parquet_path = os.path.join(processed_root, "nyc_taxi_trips")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Processed Parquet not found at: {parquet_path}")
    df = spark.read.parquet(parquet_path)
    return df


def generate_us_holidays_2013(spark: SparkSession) -> DataFrame:
    """
    Генерує базовий перелік федеральних свят США за 2013 рік.
    """
    rows = [
        ("2013-01-01", 1, "New Year's Day"),
        ("2013-01-21", 1, "Martin Luther King Jr. Day"),
        ("2013-02-18", 1, "Presidents' Day"),
        ("2013-05-27", 1, "Memorial Day"),
        ("2013-07-04", 1, "Independence Day"),
        ("2013-09-02", 1, "Labor Day"),
        ("2013-10-14", 1, "Columbus Day"),
        ("2013-11-11", 1, "Veterans Day"),
        ("2013-11-28", 1, "Thanksgiving Day"),
        ("2013-12-25", 1, "Christmas Day"),
    ]
    df = spark.createDataFrame(rows, schema="date string, is_holiday int, holiday_name string")
    return df.select(to_date(col("date")).alias("date"), col("is_holiday"), col("holiday_name"))


def load_or_prepare_weather(
    spark: SparkSession,
    processed_root: str = "/data/processed",
    csv_path: str | None = None,
) -> DataFrame | None:
    """
    Завантажує погоду з Parquet, або читає CSV і зберігає у Parquet для подальшого використання.
    Очікувана схема CSV:
      - DATE (UTC, timestamp-like string)
      - temperature_C, dewpoint_C, wind_speed_ms, precip_mm_per_hr (double)
    """
    ref_dir = os.path.join(processed_root, "reference")
    parquet_dir = os.path.join(ref_dir, "weather_hourly_utc")
    if os.path.exists(parquet_dir):
        return spark.read.parquet(parquet_dir)
    # Try explicit path or fallback to default mount
    candidate_path = csv_path
    if candidate_path is None or not os.path.exists(candidate_path):
        default_csv = os.path.join("/data/nyc", "weather", "hourly_2013_prep.csv")
        candidate_path = default_csv if os.path.exists(default_csv) else None
    if candidate_path is None:
        return None
    os.makedirs(ref_dir, exist_ok=True)
    schema = "DATE string, temperature_C double, dewpoint_C double, wind_speed_ms double, precip_mm_per_hr double"
    df_csv = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .schema(schema)
        .csv(candidate_path)
    )
    df_norm = (
        df_csv.withColumn("DATE_ts", to_timestamp(col("DATE")))
        .withColumn("timestamp_hour_utc", date_trunc("hour", col("DATE_ts")))
        .select(
            col("timestamp_hour_utc"),
            col("temperature_C"),
            col("dewpoint_C"),
            col("wind_speed_ms"),
            col("precip_mm_per_hr"),
        )
    )
    df_norm.coalesce(1).write.mode("overwrite").parquet(parquet_dir)
    return spark.read.parquet(parquet_dir)
