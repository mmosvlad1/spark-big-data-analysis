from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    when,
    year,
    month,
    dayofmonth,
    lit,
    hour,
    dayofweek,
    to_date,
    date_trunc,
    to_utc_timestamp,
    avg,
)


def add_time_features(df: DataFrame) -> DataFrame:
    with_hour = df.withColumn("pickup_hour", hour(col("pickup_datetime")))
    with_dow = with_hour.withColumn("pickup_dow", dayofweek(col("pickup_datetime")))
    return with_dow


def add_derived_metrics(df: DataFrame) -> DataFrame:
    duration_min = when(col("trip_time_in_secs") > 0, col("trip_time_in_secs") / lit(60.0))
    avg_speed = when(duration_min > 0, col("trip_distance") / (duration_min / lit(60.0)))
    cost_per_km = when(col("trip_distance") > 0, col("total_amount") / col("trip_distance"))
    cost_per_min = when(duration_min > 0, col("total_amount") / duration_min)
    return (
        df.withColumn("trip_duration_min", duration_min)
          .withColumn("avg_speed_kmh", avg_speed)
          .withColumn("cost_per_km", cost_per_km)
          .withColumn("cost_per_min", cost_per_min)
    )


def enrich_with_time_flags(df: DataFrame) -> DataFrame:
    weekend = when(col("pickup_dow").isin([1, 7]), lit(1)).otherwise(lit(0))
    day_part = when(col("pickup_hour").between(0, 5), lit("night")).otherwise(lit("day"))
    evening_peak = when(col("pickup_hour").between(16, 19), lit(1)).otherwise(lit(0))
    return (
        df.withColumn("is_weekend", weekend)
          .withColumn("day_part", day_part)
          .withColumn("is_evening_peak", evening_peak)
    )


def enrich_with_holidays(df: DataFrame, holidays_df: DataFrame) -> DataFrame:
    df_with_date = df.withColumn("pickup_date", to_date(col("pickup_datetime")))
    joined = (
        df_with_date.join(
            holidays_df.select(col("date").alias("h_date"), col("is_holiday")),
            df_with_date["pickup_date"] == col("h_date"),
            "left",
        )
        .drop("h_date")
    )
    return joined.withColumn("is_holiday", when(col("is_holiday").isNull(), lit(0)).otherwise(col("is_holiday")))


def enrich_with_borough(df: DataFrame) -> DataFrame:
    lon = col("pickup_longitude")
    lat = col("pickup_latitude")
    pickup_borough = (
        when((lon.between(lit(-74.03), lit(-73.93))) & (lat.between(lit(40.70), lit(40.88))), lit("Manhattan"))
        .when((lon.between(lit(-74.04), lit(-73.85))) & (lat.between(lit(40.57), lit(40.73))), lit("Brooklyn"))
        .when((lon.between(lit(-73.96), lit(-73.70))) & (lat.between(lit(40.54), lit(40.80))), lit("Queens"))
        .when((lon.between(lit(-73.93), lit(-73.77))) & (lat.between(lit(40.79), lit(40.91))), lit("Bronx"))
        .when((lon.between(lit(-74.26), lit(-74.05))) & (lat.between(lit(40.49), lit(40.65))), lit("Staten Island"))
        .otherwise(lit("Unknown"))
    )
    return df.withColumn("pickup_borough", pickup_borough)


def enrich_with_weather(df: DataFrame, weather_df: DataFrame) -> DataFrame:
    with_utc = df.withColumn("pickup_hour_utc", date_trunc("hour", to_utc_timestamp(col("pickup_datetime"), lit("America/New_York"))))
    joined = with_utc.join(
        weather_df.alias("w"),
        on=with_utc["pickup_hour_utc"] == col("w.timestamp_hour_utc"),
        how="left",
    ).drop("timestamp_hour_utc")
    precip = col("precip_mm_per_hr")
    wind = col("wind_speed_ms")
    precip_bin = (
        when(precip.isNull() | (precip <= lit(0.0)), lit("0"))
        .when((precip > lit(0.0)) & (precip <= lit(1.0)), lit("(0,1]"))
        .when((precip > lit(1.0)) & (precip <= lit(4.0)), lit("(1,4]"))
        .when((precip > lit(4.0)) & (precip <= lit(10.0)), lit("(4,10]"))
        .otherwise(lit(">10"))
    )
    wind_bin = (
        when(wind.isNull() | (wind < lit(3.0)), lit("[0,3)"))
        .when((wind >= lit(3.0)) & (wind < lit(7.0)), lit("[3,7)"))
        .when((wind >= lit(7.0)) & (wind < lit(12.0)), lit("[7,12)"))
        .otherwise(lit(">=12"))
    )
    temp_c = col("temperature_C")
    weather_condition_5 = (
        when(precip.isNull(), lit("unknown"))
        .when(precip <= lit(0.0), lit("clear"))
        .when((temp_c <= lit(0.5)) & (precip > lit(0.2)), lit("snow"))
        .when((precip > lit(0.0)) & (precip <= lit(1.0)), lit("drizzle"))
        .when((precip > lit(1.0)) & (precip <= lit(4.0)), lit("rain"))
        .otherwise(lit("heavy_rain"))
    )
    return (
        joined.withColumn("precip_bin", precip_bin)
              .withColumn("wind_bin", wind_bin)
              .withColumn("weather_condition_5", weather_condition_5)
    )



