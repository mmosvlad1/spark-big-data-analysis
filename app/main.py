import argparse
import os
from pyspark.sql import SparkSession

# Import functions from our processing module
from processor import (
    process_trip_data,
    process_fare_data,
    join_data,
    save_data,
    save_unmatched_data,
    print_summary_report,
    run_analytics,
    run_business_questions,
)
# Import the data loading functions from your provided loader
from data_loader import load_trip_data, load_fare_data, load_processed_trips, generate_us_holidays_2013, load_or_prepare_weather
from ml_pipeline import run_ml_pipeline


def main():
    """
    Main function to run the ETL process.
    """
    parser = argparse.ArgumentParser(description="NYC Taxi Data ETL Processor")
    # CORRECTED: Default data_root now points to /data/nyc to match docker-compose.yml
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "/data/nyc"),
        help="Path to the root directory with input data inside the container.",
    )
    # CORRECTED: Default output_root now points to /data to match docker-compose.yml
    parser.add_argument(
        "--output-root",
        default=os.environ.get("OUTPUT_ROOT", "/data"),
        help="Path to the root directory for all output data.",
    )
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="If set, skip CSV ingestion and operate only on existing processed Parquet.",
    )
    parser.add_argument(
        "--weather-path",
        default=os.environ.get("WEATHER_CSV_PATH"),
        help="Optional path to hourly weather CSV (UTC). If not provided, will try processed/reference/weather_hourly_utc.",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="If set, train ML models (Regression & Classification) on processed data.",
    )
    args = parser.parse_args()

    # --- Setup SparkSession with dynamic partition overwrite support ---
    spark = (
        SparkSession.builder.appName("NYCTaxiDataETL")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "96"))
        .getOrCreate()
    )
    
    print("SparkSession created successfully with dynamic partition overwrite enabled.")

    # --- Define paths based on corrected output root ---
    processed_path = os.path.join(args.output_root, "processed")
    quarantine_path = os.path.join(args.output_root, "quarantine")
    
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(quarantine_path, exist_ok=True)
    analytics_path = os.path.join(processed_path, "analytics")
    os.makedirs(analytics_path, exist_ok=True)
    questions_path = os.path.join(analytics_path, "questions")
    os.makedirs(questions_path, exist_ok=True)

    try:
        if args.parquet_only:
            # --- PARQUET-ONLY MODE ---
            print("\nParquet-only mode: loading processed Parquet data...")
            matched_df = load_processed_trips(spark, processed_root=processed_path)
            
            # Note: Operating on full dataset without limit
            
            matched_df.printSchema()
            matched_df.show(5, truncate=False)
            print("Parquet-only mode: processed data loaded. Further analytics will operate on this DataFrame.")
            print("\nRunning analytics on processed data...")
            run_analytics(spark, matched_df, analytics_path)
            print(f"Analytics saved under: {analytics_path}")
            print("\nRunning business questions...")
            holidays_df = generate_us_holidays_2013(spark)
            weather_df = load_or_prepare_weather(spark, processed_root=processed_path, csv_path=args.weather_path)
            if weather_df is None:
                print("Note: Weather data not found (no Parquet and no CSV). Q5 will be skipped.")
            run_business_questions(spark, matched_df, questions_path, holidays_df=holidays_df, weather_df=weather_df)
            print(f"Business question outputs saved under: {questions_path}")

            if args.train_ml:
                print("\n" + "="*50)
                print("===      ML TRAINING      ===")
                print("="*50)
                ml_output_path = os.path.join(processed_path, "models")
                os.makedirs(ml_output_path, exist_ok=True)
                run_ml_pipeline(spark, matched_df, ml_output_path)
        else:
            # --- STAGE 1: Load Data ---
            print("\Stage 1/5: Loading raw data...")
            # Use the functions from data_loader.py
            raw_trip_df = load_trip_data(spark, data_root=args.data_root)
            raw_fare_df = load_fare_data(spark, data_root=args.data_root)
            print("✔️ Data loaded successfully.")

            # --- STAGE 2: Validate and Clean Data ---
            print("\Stage 2/5: Validating and cleaning data...")
            good_trips_df, bad_trips_df, trip_stats = process_trip_data(raw_trip_df)
            good_fares_df, bad_fares_df, fare_stats = process_fare_data(raw_fare_df)
            print("✔️ Validation complete.")

            # --- STAGE 3: Join Data ---
            print("\Stage 3/5: Joining valid data...")
            (
                matched_df,
                trips_without_fare,
                fares_without_trip,
                join_stats,
            ) = join_data(good_trips_df, good_fares_df)
            print("Data joined successfully.")

            # --- STAGE 4: Save Quarantined and Unmatched Data ---
            print("\Stage 4/5: Saving data to quarantine...")
            save_data(bad_trips_df, os.path.join(quarantine_path, "invalid_trips.csv"))
            save_data(bad_fares_df, os.path.join(quarantine_path, "invalid_fares.csv"))
            save_unmatched_data(
                trips_without_fare,
                fares_without_trip,
                os.path.join(quarantine_path, "unmatched_data.csv"),
            )
            print(f"Quarantined data saved to {quarantine_path}")

            # --- STAGE 5: Enrich and Save Main Data ---
            print("\Stage 5/5: Saving processed data to Parquet...")
            save_data(matched_df, os.path.join(processed_path, "nyc_taxi_trips"), is_parquet=True)
            print(f"Main data saved to {processed_path}")

            # --- FINAL STAGE: Print Summary Report ---
            print("\nGenerating final summary report...")
            print_summary_report(spark, trip_stats, fare_stats, join_stats)

            # --- ANALYTICS: Compute and Save KPIs ---
            print("\nRunning analytics on newly processed data...")
            run_analytics(spark, matched_df, analytics_path)
            print(f"Analytics saved under: {analytics_path}")
            print("\nRunning business questions...")
            holidays_df = generate_us_holidays_2013(spark)
            weather_df = load_or_prepare_weather(spark, processed_root=processed_path, csv_path=args.weather_path)
            if weather_df is None:
                print("Note: Weather data not found (no Parquet and no CSV). Q5 will be skipped.")
            run_business_questions(spark, matched_df, questions_path, holidays_df=holidays_df, weather_df=weather_df)
            print(f"Business question outputs saved under: {questions_path}")

            if args.train_ml:
                print("\n" + "="*50)
                print("===      ML TRAINING      ===")
                print("="*50)
                ml_output_path = os.path.join(processed_path, "models")
                os.makedirs(ml_output_path, exist_ok=True)
                run_ml_pipeline(spark, matched_df, ml_output_path)

    finally:
        # --- Stop SparkSession ---
        print("\nStopping Spark session...")
        spark.stop()
        print("ETL process completed successfully!")


if __name__ == "__main__":
    main()

