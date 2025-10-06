from pyspark.sql import SparkSession
import argparse
import os

from data_loader import load_trip_data, load_fare_data, validate_dataframes


def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Data Loader")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load only one file from each category for testing",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "/data/nyc"),
        help="Path to dataset root mounted in container",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.appName("NYCTaxiDataLoad").getOrCreate()
    print("SparkSession created successfully!")

    trip_df = load_trip_data(spark, data_root=args.data_root, test=args.test)
    fare_df = load_fare_data(spark, data_root=args.data_root, test=args.test)

    validate_dataframes(trip_df, fare_df)

    spark.stop()
    print("Spark job finished successfully.")


if __name__ == "__main__":
    main()
