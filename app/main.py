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
)
# Import the data loading functions from your provided loader
from data_loader import load_trip_data, load_fare_data


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
    args = parser.parse_args()

    # --- Setup SparkSession with dynamic partition overwrite support ---
    spark = (
        SparkSession.builder.appName("NYCTaxiDataETL")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    
    print("SparkSession created successfully with dynamic partition overwrite enabled.")

    # --- Define paths based on corrected output root ---
    processed_path = os.path.join(args.output_root, "processed")
    quarantine_path = os.path.join(args.output_root, "quarantine")
    
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(quarantine_path, exist_ok=True)

    try:
        # --- STAGE 1: Load Data ---
        print("\n🚀 Stage 1/5: Loading raw data...")
        # Use the functions from data_loader.py
        raw_trip_df = load_trip_data(spark, data_root=args.data_root)
        raw_fare_df = load_fare_data(spark, data_root=args.data_root)
        print("✔️ Data loaded successfully.")

        # --- STAGE 2: Validate and Clean Data ---
        print("\n🚀 Stage 2/5: Validating and cleaning data...")
        good_trips_df, bad_trips_df, trip_stats = process_trip_data(raw_trip_df)
        good_fares_df, bad_fares_df, fare_stats = process_fare_data(raw_fare_df)
        print("✔️ Validation complete.")

        # --- STAGE 3: Join Data ---
        print("\n🚀 Stage 3/5: Joining valid data...")
        (
            matched_df,
            trips_without_fare,
            fares_without_trip,
            join_stats,
        ) = join_data(good_trips_df, good_fares_df)
        print("✔️ Data joined successfully.")

        # --- STAGE 4: Save Quarantined and Unmatched Data ---
        print("\n🚀 Stage 4/5: Saving data to quarantine...")
        save_data(bad_trips_df, os.path.join(quarantine_path, "invalid_trips.csv"))
        save_data(bad_fares_df, os.path.join(quarantine_path, "invalid_fares.csv"))
        save_unmatched_data(
            trips_without_fare,
            fares_without_trip,
            os.path.join(quarantine_path, "unmatched_data.csv"),
        )
        print(f"✔️ Quarantined data saved to {quarantine_path}")

        # --- STAGE 5: Enrich and Save Main Data ---
        print("\n🚀 Stage 5/5: Saving processed data to Parquet...")
        save_data(matched_df, os.path.join(processed_path, "nyc_taxi_trips"), is_parquet=True)
        print(f"✔️ Main data saved to {processed_path}")

        # --- FINAL STAGE: Print Summary Report ---
        print("\nGenerating final summary report...")
        print_summary_report(spark, trip_stats, fare_stats, join_stats)

    finally:
        # --- Stop SparkSession ---
        print("\n🏁 Stopping Spark session...")
        spark.stop()
        print("✅ ETL process completed successfully!")


if __name__ == "__main__":
    main()

