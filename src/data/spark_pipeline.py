from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType, TimestampType
from pyspark.sql.window import Window

from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_spark_session(app_name: str = "InfraInsightDataPlatform") -> SparkSession:
    """Initialisiert eine lokale, für Apple Silicon optimierte Spark Session."""
    logger.info("Initialisiere Apache Spark Session...")
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")  # Optimiert für lokale M3 Cores
        .master("local[*]")
        .getOrCreate()
    )


def run_spark_ingestion(config: Config):
    """Phase 1: Lädt Rohdaten und schreibt sie in den Data Vault 2.0 Layer."""
    spark = create_spark_session()

    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / config.paths["raw_data_path"]
    vault_dir = project_root / "data" / "data_vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lade Rohdaten via Spark von: {raw_path}")

    raw_schema = StructType(
        [StructField("timestamp", TimestampType(), True), StructField("value", DoubleType(), True)]
    )

    df_raw = spark.read.csv(str(raw_path), header=True, schema=raw_schema)

    SRC_INSTANCE_ID = "i-09ab12cd34ef5678a"
    RECORD_SOURCE = "AWS_EC2_METRICS"

    df_vault = (
        df_raw.withColumn("instance_id", F.lit(SRC_INSTANCE_ID))
        .withColumn("record_source", F.lit(RECORD_SOURCE))
        .withColumn("load_timestamp", F.current_timestamp())
        .withColumn("hk_server", F.sha2(F.upper(F.trim(F.col("instance_id"))), 256))
        .withColumn("hash_diff", F.sha2(F.concat_ws(";", F.col("value")), 256))
    )

    hub_path = vault_dir / "hub_server"
    sat_path = vault_dir / "sat_server_telemetry"

    df_vault.select(
        "hk_server", "instance_id", "load_timestamp", "record_source"
    ).distinct().write.mode("overwrite").parquet(str(hub_path))
    df_vault.select(
        "hk_server",
        "timestamp",
        "load_timestamp",
        "hash_diff",
        F.col("value").alias("cpu_utilization"),
        "record_source",
    ).write.mode("overwrite").parquet(str(sat_path))

    logger.info("✅ Data Vault 2.0 Tabellen erfolgreich geschrieben.")
    spark.stop()


def run_spark_feature_engineering():
    """Phase 2: Liest Daten aus dem Data Vault Satelliten und berechnet

    multivariate Zeitreihen-Features via Spark Window-Functions.
    """
    spark = create_spark_session("InfraInsightFeatureEngineering")

    project_root = Path(__file__).resolve().parents[2]
    sat_path = project_root / "data" / "data_vault" / "sat_server_telemetry"
    processed_dir = project_root / "data" / "processed_spark"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lade Satellitendaten für Feature Engineering aus: {sat_path}")
    df_sat = spark.read.parquet(str(sat_path))

    # Spark Window-Spezifikation definieren (Chronologisch sortiert nach Server)
    window_1h = Window.partitionBy("hk_server").orderBy("timestamp").rowsBetween(-11, 0)
    window_6h = Window.partitionBy("hk_server").orderBy("timestamp").rowsBetween(-71, 0)

    logger.info(
        "Berechne zyklische Zeit-Features (Stunde & Wochentag) und rollierende Statistiken..."
    )

    PI = 3.141592653589793

    # Verteiltes Transformation-Query
    df_features = (
        df_sat
        # A. Zyklische Zeit-Features: Stunde des Tages (24h)
        .withColumn("hour", F.hour(F.col("timestamp")))
        .withColumn("hour_sin", F.sin(2 * F.lit(PI) * F.col("hour") / 24.0))
        .withColumn("hour_cos", F.cos(2 * F.lit(PI) * F.col("hour") / 24.0))
        # B. Zyklische Zeit-Features: Wochentag (7 Tage) - NEU HINZUGEFÜGT
        .withColumn("dayofweek", F.dayofweek(F.col("timestamp")))
        .withColumn("day_sin", F.sin(2 * F.lit(PI) * F.col("dayofweek") / 7.0))
        .withColumn("day_cos", F.cos(2 * F.lit(PI) * F.col("dayofweek") / 7.0))
        # C. Rollierende Metriken via Window Functions
        .withColumn("rolling_mean_1h", F.avg("cpu_utilization").over(window_1h))
        .withColumn("rolling_std_1h", F.stddev("cpu_utilization").over(window_1h))
        .withColumn("rolling_mean_6h", F.avg("cpu_utilization").over(window_6h))
        .withColumn("rolling_std_6h", F.stddev("cpu_utilization").over(window_6h))
        # Strategisches Bereinigen unvollständiger Fenster am Anfang
        .dropna()
    )

    # In-Memory-Performance optimieren für nachfolgende Aktionen
    df_features.cache()

    # --- PERFORMANCE ANALYSE (DoD Erfüllung) ---
    logger.info("📊 Generiere Spark Execution Plan (Physischer Ausführungsplan):")
    df_features.explain(mode="formatted")

    # Features abspeichern
    output_path = processed_dir / "features.parquet"
    df_features.write.mode("overwrite").parquet(str(output_path))

    logger.info(
        f"✅ Multivariate Features inkl. Wochentag erfolgreich unter '{output_path}' persistiert."
    )

    # Cache wieder freigeben
    df_features.unpersist()
    spark.stop()


if __name__ == "__main__":
    config = Config()
    run_spark_ingestion(config)
    run_spark_feature_engineering()
