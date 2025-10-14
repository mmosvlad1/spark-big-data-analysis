from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql.functions import col, when, year, month, dayofmonth, lit


def process_trip_data(df: DataFrame) -> (DataFrame, DataFrame, dict):
    """
    Виконує повний цикл очищення та валідації для даних поїздок.
    Повертає валідні дані, невалідні дані та словник зі статистикою.
    """
    total_count = df.count()
    
    # 1. Видалення дублікатів
    key_cols = ["medallion", "hack_license", "vendor_id", "pickup_datetime"]
    df_no_duplicates = df.dropDuplicates(subset=key_cols)
    
    # 2. Заповнення пропусків
    df_filled = df_no_duplicates.fillna({
        "rate_code": "UNK",
        "store_and_fwd_flag": "N"
    })
    
    # 3. Категоризація викидів та помилок
    rejection_reason_col = (
        when(col("passenger_count") <= 0, "invalid_passenger_count")
        .when(col("passenger_count") > 6, "excessive_passenger_count")
        .when(col("trip_time_in_secs") < 60, "trip_too_short")
        .when(col("trip_time_in_secs") > 43200, "trip_too_long")
        .when(col("trip_distance") <= 0, "zero_or_negative_distance")
        .when(col("trip_distance") > 200, "excessive_distance")
        .when(col("pickup_datetime") >= col("dropoff_datetime"), "invalid_trip_duration")
        .otherwise(lit("VALID"))
    )
    df_with_reasons = df_filled.withColumn("rejection_reason", rejection_reason_col)

    # 4. Розділення на валідні та невалідні дані
    good_df = df_with_reasons.filter(col("rejection_reason") == "VALID").drop("rejection_reason")
    bad_df = df_with_reasons.filter(col("rejection_reason") != "VALID")

    # 5. Збір статистики
    bad_count = bad_df.count()
    stats = {
        "total_records": total_count,
        "rejected_records": bad_count,
        "rejection_reasons": bad_df.groupBy("rejection_reason").count().collect() if bad_count > 0 else []
    }
    
    return good_df, bad_df, stats


def process_fare_data(df: DataFrame) -> (DataFrame, DataFrame, dict):
    """
    Виконує повний цикл очищення та валідації для даних оплат.
    Повертає валідні дані, невалідні дані та словник зі статистикою.
    """
    total_count = df.count()

    # 1. Видалення дублікатів
    key_cols = ["medallion", "hack_license", "vendor_id", "pickup_datetime"]
    df_no_duplicates = df.dropDuplicates(subset=key_cols)

    # 2. Категоризація викидів
    rejection_reason_col = (
        when(col("total_amount") < 2.5, "amount_too_low")
        .when(col("total_amount") > 1000, "amount_too_high")
        .when(col("fare_amount") < 0, "negative_fare_amount")
        .otherwise(lit("VALID"))
    )
    df_with_reasons = df_no_duplicates.withColumn("rejection_reason", rejection_reason_col)

    # 3. Розділення на валідні та невалідні дані
    good_df = df_with_reasons.filter(col("rejection_reason") == "VALID").drop("rejection_reason")
    bad_df = df_with_reasons.filter(col("rejection_reason") != "VALID")

    # 4. Збір статистики
    bad_count = bad_df.count()
    stats = {
        "total_records": total_count,
        "rejected_records": bad_count,
        "rejection_reasons": bad_df.groupBy("rejection_reason").count().collect() if bad_count > 0 else []
    }
    
    return good_df, bad_df, stats


def join_data(trip_df: DataFrame, fare_df: DataFrame) -> (DataFrame, DataFrame, DataFrame, dict):
    """
    Об'єднує дані поїздок та оплат, розділяючи результат на 3 потоки.
    Повертає об'єднані дані та словник зі статистикою.
    """
    trip_count = trip_df.count()
    fare_count = fare_df.count()
    
    join_keys = ["medallion", "hack_license", "vendor_id", "pickup_datetime"]
    joined_df = trip_df.alias("t").join(fare_df.alias("f"), join_keys, "full_outer")

    matched_df = joined_df.filter(col("t.dropoff_datetime").isNotNull() & col("f.total_amount").isNotNull())
    trips_without_fare = joined_df.filter(col("t.dropoff_datetime").isNotNull() & col("f.total_amount").isNull()).select("t.*")
    fares_without_trip = joined_df.filter(col("t.dropoff_datetime").isNull() & col("f.total_amount").isNotNull()).select("f.*")

    # Статистика об'єднання
    unmatched_trips_count = trips_without_fare.count()
    unmatched_fares_count = fares_without_trip.count()

    stats = {
        "valid_trips_for_join": trip_count,
        "valid_fares_for_join": fare_count,
        "trips_without_fare": unmatched_trips_count,
        "fares_without_trip": unmatched_fares_count,
        "matched_records": matched_df.count()
    }

    return matched_df, trips_without_fare, fares_without_trip, stats


def save_data(df: DataFrame, path: str, is_parquet: bool = False):
    """
    Зберігає DataFrame у CSV або Parquet.
    """
    if is_parquet:
        df_to_save = df.withColumn("year", year(col("pickup_datetime"))) \
                       .withColumn("month", month(col("pickup_datetime"))) \
                       .withColumn("day", dayofmonth(col("pickup_datetime")))
        df_to_save.coalesce(1).write.partitionBy("year", "month", "day").mode("overwrite").parquet(path)
    else:
        df.coalesce(1).write.option("header", True).mode("overwrite").csv(path)


def save_unmatched_data(trips_df: DataFrame, fares_df: DataFrame, path: str):
    """
    Об'єднує та зберігає незіставлені дані в один CSV файл.
    """
    trips_df = trips_df.withColumn("source", lit("trip_data"))
    fares_df = fares_df.withColumn("source", lit("fare_data"))
    unmatched_df = trips_df.unionByName(fares_df, allowMissingColumns=True)
    unmatched_df.coalesce(1).write.option("header", True).mode("overwrite").csv(path)


def print_summary_report(spark: SparkSession, trip_stats: dict, fare_stats: dict, join_stats: dict):
    """
    Друкує фінальний звіт зі статистикою обробки.
    """
    print("\n" + "="*50)
    print("===      ЗВІТ ПРО ОБРОБКУ ДАНИХ      ===")
    print("="*50)

    # --- Статистика поїздок (Trip Data) ---
    total = trip_stats['total_records']
    bad = trip_stats['rejected_records']
    percentage = (bad / total * 100) if total > 0 else 0
    print("\n--- Обробка Trip Data ---")
    print(f"  > Всього записів: {total}")
    print(f"  > Відбраковано: {bad} ({percentage:.2f}%)")
    if bad > 0:
        print("  > Причини відбраковки:")
        reasons_df = spark.createDataFrame(trip_stats['rejection_reasons'])
        reasons_df.show(truncate=False)

    # --- Статистика оплат (Fare Data) ---
    total = fare_stats['total_records']
    bad = fare_stats['rejected_records']
    percentage = (bad / total * 100) if total > 0 else 0
    print("\n--- Обробка Fare Data ---")
    print(f"  > Всього записів: {total}")
    print(f"  > Відбраковано: {bad} ({percentage:.2f}%)")
    if bad > 0:
        print("  > Причини відбраковки:")
        reasons_df = spark.createDataFrame(fare_stats['rejection_reasons'])
        reasons_df.show(truncate=False)

    # --- Статистика об'єднання (Join) ---
    trips_in = join_stats['valid_trips_for_join']
    fares_in = join_stats['valid_fares_for_join']
    trips_out = join_stats['trips_without_fare']
    fares_out = join_stats['fares_without_trip']
    
    trip_perc = (trips_out / trips_in * 100) if trips_in > 0 else 0
    fare_perc = (fares_out / fares_in * 100) if fares_in > 0 else 0
    
    print("\n--- Статистика об'єднання ---")
    print(f"  > Валідних поїздок для об'єднання: {trips_in}")
    print(f"  > Валідних оплат для об'єднання: {fares_in}")
    print(f"  > Успішно об'єднано: {join_stats['matched_records']}")
    print(f"  > Поїздки без оплати: {trips_out} ({trip_perc:.2f}%)")
    print(f"  > Оплати без поїздки: {fares_out} ({fare_perc:.2f}%)")

    print("\n" + "="*50)