from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql.functions import (
    col,
    when,
    lit,
    avg,
    sum as sum_,
    stddev,
    count,
    expr,
)
import os
import json
from datetime import datetime

from enrichments import add_time_features, add_derived_metrics, enrich_with_time_flags, enrich_with_holidays, enrich_with_borough, enrich_with_weather


def save_json_text(spark: SparkSession, obj: dict, path: str) -> None:
    json_str = json.dumps(obj, ensure_ascii=False)
    spark.createDataFrame([json_str], "string").coalesce(1).write.mode("overwrite").text(path)


def _format_value(val, dtype: str = "default") -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, float):
        if dtype == "money":
            return f"{val:.2f}"
        elif dtype == "count":
            return str(int(val))
        else:
            return f"{val:.4f}"
    return str(val)


def build_analytics_summary_csv(overview: dict, numeric_stats_df: DataFrame, kpi_dfs: dict) -> str:
    lines = []
    lines.append("# NYC Taxi Analytics Summary Report")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append(f"# Total records: {overview['row_count']}")
    if overview['min_pickup_datetime']:
        lines.append(f"# Date range: {overview['min_pickup_datetime']} to {overview['max_pickup_datetime']}")
    lines.append("")
    lines.append("## NUMERIC_STATISTICS")
    numeric_rows = numeric_stats_df.collect()
    if numeric_rows:
        lines.append("column,count,null_count,min,max,mean,stddev,sum,p01,p05,p50,p95,p99")
        for row in numeric_rows:
            values = [
                row["column"],
                str(row["count"]),
                str(row["null_count"]),
                _format_value(row["min"]),
                _format_value(row["max"]),
                _format_value(row["mean"]),
                _format_value(row["stddev"]),
                _format_value(row["sum"], "money"),
                _format_value(row["p01"]),
                _format_value(row["p05"]),
                _format_value(row["p50"]),
                _format_value(row["p95"]),
                _format_value(row["p99"]),
            ]
            lines.append(",".join(values))
    lines.append("")
    for section_name in ["kpi_by_hour", "kpi_by_dow", "kpi_by_year_month", "kpi_by_payment", "kpi_by_vendor", "kpi_by_passengers"]:
        if section_name not in kpi_dfs:
            continue
        lines.append(f"## {section_name.upper()}")
        kpi_df = kpi_dfs[section_name]
        kpi_rows = kpi_df.collect()
        if kpi_rows:
            col_names = list(kpi_rows[0].asDict().keys())
            lines.append(",".join(col_names))
            for row in kpi_rows:
                values = []
                for c in col_names:
                    v = row[c]
                    if "amount" in c or "cost" in c or "revenue" in c:
                        values.append(_format_value(v, "money"))
                    elif c in ["trips", "count"]:
                        values.append(_format_value(v, "count"))
                    else:
                        values.append(_format_value(v))
                lines.append(",".join(values))
        lines.append("")
    return "\n".join(lines)


def compute_dataset_overview(spark: SparkSession, df: DataFrame) -> dict:
    try:
        agg_result = df.agg(
            count(lit(1)).alias("row_count"),
            expr("min(pickup_datetime)").alias("min_pickup_datetime"),
            expr("max(pickup_datetime)").alias("max_pickup_datetime"),
        ).collect()[0]
        row_count = int(agg_result["row_count"]) if agg_result["row_count"] is not None else 0
        min_dt = str(agg_result["min_pickup_datetime"]) if agg_result["min_pickup_datetime"] else None
        max_dt = str(agg_result["max_pickup_datetime"]) if agg_result["max_pickup_datetime"] else None
    except Exception:
        row_count, min_dt, max_dt = 0, None, None
    schema_fields = [(name, dtype) for name, dtype in df.dtypes]
    return {
        "row_count": row_count,
        "num_columns": len(schema_fields),
        "schema": [{"name": name, "type": dtype} for name, dtype in schema_fields],
        "min_pickup_datetime": min_dt,
        "max_pickup_datetime": max_dt,
    }


def compute_numeric_stats(spark: SparkSession, df: DataFrame) -> DataFrame:
    numeric_types = {"int", "bigint", "double", "float", "long", "short", "decimal"}
    numeric_cols = [name for name, dtype in df.dtypes if any(t in dtype for t in numeric_types)]
    prefer = [
        "trip_time_in_secs",
        "trip_distance",
        "passenger_count",
        "fare_amount",
        "surcharge",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "total_amount",
        "trip_duration_min",
        "avg_speed_kmh",
        "cost_per_km",
        "cost_per_min",
    ]
    numeric_cols = [c for c in prefer if c in numeric_cols]
    if not numeric_cols:
        return spark.createDataFrame([], schema="column string, count long, null_count long, min double, max double, mean double, stddev double, sum double, p01 double, p05 double, p50 double, p95 double, p99 double")
    agg_exprs = []
    for c in numeric_cols:
        agg_exprs.append(count(col(c)).alias(f"{c}__count_non_null"))
        agg_exprs.append(sum_(col(c)).alias(f"{c}__sum"))
        agg_exprs.append(avg(col(c)).alias(f"{c}__mean"))
        agg_exprs.append(stddev(col(c)).alias(f"{c}__stddev"))
        agg_exprs.append(expr(f"min({c})").alias(f"{c}__min"))
        agg_exprs.append(expr(f"max({c})").alias(f"{c}__max"))
        agg_exprs.append(sum_(when(col(c).isNull(), 1).otherwise(0)).alias(f"{c}__null_count"))
    base_metrics_row = df.agg(*agg_exprs).collect()[0]
    probs = [0.01, 0.05, 0.5, 0.95, 0.99]
    rel_error = 0.05
    quantiles = df.stat.approxQuantile(numeric_cols, probs, rel_error)
    rows = []
    for idx, c in enumerate(numeric_cols):
        rows.append(Row(**{
            "column": c,
            "count": int(base_metrics_row[f"{c}__count_non_null"]) if base_metrics_row[f"{c}__count_non_null"] is not None else 0,
            "null_count": int(base_metrics_row[f"{c}__null_count"]) if base_metrics_row[f"{c}__null_count"] is not None else 0,
            "min": float(base_metrics_row[f"{c}__min"]) if base_metrics_row[f"{c}__min"] is not None else None,
            "max": float(base_metrics_row[f"{c}__max"]) if base_metrics_row[f"{c}__max"] is not None else None,
            "mean": float(base_metrics_row[f"{c}__mean"]) if base_metrics_row[f"{c}__mean"] is not None else None,
            "stddev": float(base_metrics_row[f"{c}__stddev"]) if base_metrics_row[f"{c}__stddev"] is not None else None,
            "sum": float(base_metrics_row[f"{c}__sum"]) if base_metrics_row[f"{c}__sum"] is not None else None,
            "p01": float(quantiles[idx][0]) if quantiles[idx][0] is not None else None,
            "p05": float(quantiles[idx][1]) if quantiles[idx][1] is not None else None,
            "p50": float(quantiles[idx][2]) if quantiles[idx][2] is not None else None,
            "p95": float(quantiles[idx][3]) if quantiles[idx][3] is not None else None,
            "p99": float(quantiles[idx][4]) if quantiles[idx][4] is not None else None,
        }))
    return spark.createDataFrame(rows)


def compute_segmented_kpis(df: DataFrame) -> dict:
    kpi_exprs = [
        avg(col("total_amount")).alias("avg_total_amount"),
        expr("percentile_approx(total_amount, 0.5)").alias("p50_total_amount"),
        expr("percentile_approx(total_amount, 0.95)").alias("p95_total_amount"),
        avg(col("cost_per_km")).alias("avg_cost_per_km"),
        avg(col("cost_per_min")).alias("avg_cost_per_min"),
        avg(col("avg_speed_kmh")).alias("avg_speed_kmh"),
        count(lit(1)).alias("trips"),
        sum_(col("total_amount")).alias("revenue"),
    ]
    out = {}
    if "pickup_hour" in df.columns:
        out["kpi_by_hour"] = df.groupBy("pickup_hour").agg(*kpi_exprs)
    if "pickup_dow" in df.columns:
        out["kpi_by_dow"] = df.groupBy("pickup_dow").agg(*kpi_exprs)
    if "year" in df.columns and "month" in df.columns:
        out["kpi_by_year_month"] = df.groupBy("year", "month").agg(*kpi_exprs)
    if "payment_type" in df.columns:
        out["kpi_by_payment"] = df.groupBy("payment_type").agg(*kpi_exprs)
    if "vendor_id" in df.columns:
        out["kpi_by_vendor"] = df.groupBy("vendor_id").agg(*kpi_exprs)
    if "passenger_count" in df.columns:
        out["kpi_by_passengers"] = df.groupBy("passenger_count").agg(*kpi_exprs)
    return out


def run_analytics(spark: SparkSession, df: DataFrame, analytics_root: str) -> None:
    os.makedirs(analytics_root, exist_ok=True)
    enriched = add_time_features(add_derived_metrics(df))
    overview = compute_dataset_overview(spark, enriched)
    save_json_text(spark, overview, os.path.join(analytics_root, "overview.json"))
    numeric_stats_df = compute_numeric_stats(spark, enriched)
    segmented = compute_segmented_kpis(enriched)
    summary_csv = build_analytics_summary_csv(overview, numeric_stats_df, segmented)
    summary_path = os.path.join(analytics_root, "analytics_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_csv)


def run_business_questions(spark: SparkSession, df: DataFrame, out_root: str, holidays_df: DataFrame = None, weather_df: DataFrame = None) -> None:
    os.makedirs(out_root, exist_ok=True)
    base = add_time_features(add_derived_metrics(df))
    base = enrich_with_time_flags(base)
    if holidays_df is not None:
        base = enrich_with_holidays(base, holidays_df)
    base = enrich_with_borough(base)
    if weather_df is not None:
        base = enrich_with_weather(base, weather_df)

    q1 = (
        base.filter((col("trip_distance") >= lit(1.0)) & (col("trip_distance") <= lit(10.0)))
            .groupBy("pickup_hour", "is_weekend")
            .agg(
                sum_(col("total_amount")).alias("revenue"),
                count(lit(1)).alias("trips"),
                avg(col("avg_speed_kmh")).alias("avg_speed_kmh"),
                avg(col("total_amount")).alias("avg_total_amount"),
            )
            .orderBy("pickup_hour", "is_weekend")
    )
    q1.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q1_hourly_weekday_weekend.csv"))

    q2 = (
        base.groupBy("payment_type", "passenger_count", "day_part")
            .agg(
                avg(col("tip_amount")).alias("avg_tip"),
                expr("percentile_approx(tip_amount, 0.5)").alias("p50_tip"),
                count(lit(1)).alias("trips"),
            )
    )
    q2.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q2_tips_by_payment_passengers_daypart.csv"))

    monthly = (
        base.groupBy("vendor_id", "year", "month")
            .agg(avg(col("cost_per_km")).alias("avg_cost_per_km"))
    )
    window_spec = "partition by vendor_id order by year, month"
    q4_mom = (
        monthly.select(
            col("vendor_id"),
            col("year"),
            col("month"),
            col("avg_cost_per_km"),
            expr(f"lag(avg_cost_per_km, 1) over ({window_spec})").alias("lag_avg"),
        )
        .withColumn("mom_change", col("avg_cost_per_km") - col("lag_avg"))
    )
    q4_mom.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q4_vendor_mom_cost_per_km.csv"))

    q4_up = q4_mom.withColumn("is_up", when(col("mom_change") > lit(0), lit(1)).otherwise(lit(0)))
    grp = expr("sum(case when is_up=0 or is_up is null then 1 else 0 end) over (partition by vendor_id order by year, month rows between unbounded preceding and current row)").alias("grp")
    q4_streaks = (
        q4_up.select("vendor_id", "year", "month", "is_up", grp)
             .withColumn("streak_len", expr("count(1) over (partition by vendor_id, grp)"))
             .where(col("is_up") == lit(1))
             .groupBy("vendor_id")
             .agg(expr("max(streak_len)").alias("longest_growth_streak_months"))
    )
    q4_streaks.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q4_vendor_longest_growth_streak.csv"))

    with_holiday = base if "is_holiday" in base.columns else base.withColumn("is_holiday", lit(0))
    q6 = (
        with_holiday.groupBy("pickup_borough", "is_holiday")
            .agg(
                count(lit(1)).alias("trips"),
                sum_(when(col("total_amount") > lit(50.0), lit(1)).otherwise(lit(0))).alias("high_value_trips"),
            )
            .withColumn("share", col("high_value_trips") / col("trips"))
    )
    q6.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q6_high_value_share_by_borough_and_holiday.csv"))

    q3_filtered = base.filter(
        (col("is_evening_peak") == lit(1)) &
        (col("trip_distance").between(lit(1.0), lit(50.0))) &
        (col("avg_speed_kmh").between(lit(5.0), lit(80.0)))
    )
    q3_borough = (
        q3_filtered.groupBy("pickup_borough")
            .agg(
                avg(col("total_amount")).alias("avg_total_amount"),
                count(lit(1)).alias("trips"),
            )
    )
    q3_borough.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q3_evening_peak_by_borough.csv"))

    if weather_df is not None and "weather_condition_5" in base.columns:
        q5a = (
            base.groupBy("weather_condition_5", "pickup_hour")
                .agg(
                    avg(col("avg_speed_kmh")).alias("avg_speed_kmh"),
                    avg(col("cost_per_min")).alias("avg_cost_per_min"),
                    count(lit(1)).alias("trips"),
                    avg(col("temperature_C")).alias("avg_temperature_C"),
                    avg(col("dewpoint_C")).alias("avg_dewpoint_C"),
                    avg(col("wind_speed_ms")).alias("avg_wind_speed_ms"),
                    avg(col("precip_mm_per_hr")).alias("avg_precip_mm_per_hr"),
                )
                .orderBy("weather_condition_5", "pickup_hour")
        )
        q5a.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q5_weather_by_condition_hour.csv"))

        q5b = (
            base.groupBy("precip_bin", "wind_bin", "pickup_hour")
                .agg(
                    avg(col("avg_speed_kmh")).alias("avg_speed_kmh"),
                    avg(col("cost_per_min")).alias("avg_cost_per_min"),
                    count(lit(1)).alias("trips"),
                    avg(col("temperature_C")).alias("avg_temperature_C"),
                    avg(col("dewpoint_C")).alias("avg_dewpoint_C"),
                    avg(col("wind_speed_ms")).alias("avg_wind_speed_ms"),
                    avg(col("precip_mm_per_hr")).alias("avg_precip_mm_per_hr"),
                )
                .orderBy("precip_bin", "wind_bin", "pickup_hour")
        )
        q5b.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_root, "q5_weather_by_precip_wind_hour.csv"))



