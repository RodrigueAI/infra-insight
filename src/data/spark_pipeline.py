import os
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, TimestampType

from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_spark_session(app_name: str = "InfraInsightDataPlatform") -> SparkSession:
    """Initialisiert eine lokale, für Apple Silicon optimierte Spark Session."""
    logger.info("Initialisiere Apache Spark Session...")
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .master("local[*]")  # Nutzt alle CPU-Kerne deines M3
        .getOrCreate()
    )
    return spark


def run_spark_ingestion(config: Config):
    """Lädt die Rohdaten über PySpark, berechnet das Data Vault 2.0 Schema

    und generiert die multivariaten Zeitreihen-Features.
    """
    spark = create_spark_session()

    # 1. Pfade auflösen
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / config.paths["raw_data_path"]
    vault_dir = project_root / "data" / "data_vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lade Rohdaten via Spark von: {raw_path}")

    # Explicit Schema Definition für Typsicherheit beim Einlesen
    raw_schema = StructType(
        [StructField("timestamp", TimestampType(), True), StructField("value", DoubleType(), True)]
    )

    # 2. Raw Ingestion
    df_raw = spark.read.csv(str(raw_path), header=True, schema=raw_schema)

    # Da wir aktuell eine Single-Server-CSV haben, fügen wir eine synthetische Instance-ID hinzu.
    # Sobald du Multi-Server-Logs hast, fällt dieser Schritt weg.
    SRC_INSTANCE_ID = "i-09ab12cd34ef5678a"
    RECORD_SOURCE = "AWS_EC2_METRICS"

    logger.info("Transformiere Rohdaten in Data Vault 2.0 Strukturen...")

    # Metadaten & Hashes injizieren via Spark SQL Functions
    df_vault = (
        df_raw.withColumn("instance_id", F.lit(SRC_INSTANCE_ID))
        .withColumn("record_source", F.lit(RECORD_SOURCE))
        .withColumn("load_timestamp", F.current_timestamp())
        # DV 2.0 Hash Key: Clean, Upper, MD5/SHA-256
        .withColumn("hk_server", F.sha2(F.upper(F.trim(F.col("instance_id"))), 256))
        # DV 2.0 Hash Diff: Bestimmt Änderungen in den Metriken (hier vorbereitet für Multivariate)
        .withColumn("hash_diff", F.sha2(F.concat_ws(";", F.col("value")), 256))
    )

    # 3. Extraktion & Speicherung der Vault-Tabellen (Parquet-Format für maximale Performance)
    hub_server = df_vault.select(
        "hk_server", "instance_id", "load_timestamp", "record_source"
    ).distinct()
    sat_telemetry = df_vault.select(
        "hk_server",
        "timestamp",
        "load_timestamp",
        "hash_diff",
        F.col("value").alias("cpu_utilization"),
        "record_source",
    )

    hub_path = vault_dir / "hub_server"
    sat_path = vault_dir / "sat_server_telemetry"

    # Revisionssicher abspeichern im Data Lake / Vault Directory
    hub_server.write.mode("overwrite").parquet(str(hub_path))
    sat_telemetry.write.mode("overwrite").parquet(str(sat_path))

    logger.info(f"💾 Hub erfolgreich persistiert unter: {hub_path}")
    logger.info(f"💾 Satellit erfolgreich persistiert unter: {sat_path}")

    spark.stop()
    logger.info("Spark Session erfolgreich geschlossen.")


if __name__ == "__main__":
    config = Config()
    run_spark_ingestion(config)
