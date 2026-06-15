from pathlib import Path

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array  # NEU: Löst den IndexError auf PyTorch-Ebene
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
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
        .config("spark.sql.shuffle.partitions", "4")
        .master("local[*]")
        .getOrCreate()
    )


def run_spark_ingestion(config: Config):
    """Phase 1: Ingestiert Rohdaten und lädt sie in den Data Vault."""
    spark = create_spark_session()
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / config.paths["raw_data_path"]
    vault_dir = project_root / "data" / "data_vault"

    raw_schema = StructType(
        [StructField("timestamp", TimestampType(), True), StructField("value", DoubleType(), True)]
    )

    df_raw = spark.read.csv(str(raw_path), header=True, schema=raw_schema)
    SRC_INSTANCE_ID = "i-09ab12cd34ef5678a"

    df_vault = (
        df_raw.withColumn("instance_id", F.lit(SRC_INSTANCE_ID))
        .withColumn("record_source_cpu", F.lit("OS_METRIC_AGENT"))
        .withColumn("record_source_net", F.lit("AWS_VPC_FLOW_LOGS"))
        .withColumn(
            "combined_source",
            F.concat_ws("+", F.col("record_source_cpu"), F.col("record_source_net")),
        )
        .withColumn("cpu_utilization", F.col("value"))
        .withColumn("network_in_mbps", F.col("value") * 1.5 + 10.0)
        .withColumn("load_timestamp", F.current_timestamp())
        .withColumn("hk_server", F.sha2(F.upper(F.trim(F.col("instance_id"))), 256))
        .withColumn(
            "hash_diff",
            F.sha2(F.concat_ws(";", F.col("cpu_utilization"), F.col("network_in_mbps")), 256),
        )
    )

    df_vault.select(
        "hk_server",
        "instance_id",
        "load_timestamp",
        F.col("combined_source").alias("record_source"),
    ).distinct().write.mode("overwrite").parquet(str(vault_dir / "hub_server"))
    df_vault.select(
        "hk_server",
        "timestamp",
        "load_timestamp",
        "hash_diff",
        "cpu_utilization",
        "network_in_mbps",
        F.col("combined_source").alias("record_source"),
    ).write.mode("overwrite").parquet(str(vault_dir / "sat_server_telemetry"))

    logger.info("✅ Data Vault 2.0 Tabellen geladen.")
    spark.stop()


def run_spark_feature_engineering(config: Config):
    """Phase 2: Berechnet Features, transformiert ML-Vektoren in primitive Arrays

    und exportiert hocheffiziente, vorsortierte chronologische Parquet-Splits.
    """
    spark = create_spark_session("InfraInsightFeatureEngineering")
    project_root = Path(__file__).resolve().parents[2]
    sat_path = project_root / "data" / "data_vault" / "sat_server_telemetry"
    processed_dir = project_root / "data" / "processed_spark"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Daten laden und zwingend chronologisch sortieren
    df_sat = spark.read.parquet(str(sat_path)).orderBy("timestamp")

    window_1h = Window.partitionBy("hk_server").orderBy("timestamp").rowsBetween(-11, 0)
    window_6h = Window.partitionBy("hk_server").orderBy("timestamp").rowsBetween(-71, 0)

    PI = 3.141592653589793

    df_features = (
        df_sat.withColumn("hour", F.hour(F.col("timestamp")))
        .withColumn("hour_sin", F.sin(2 * F.lit(PI) * F.col("hour") / 24.0))
        .withColumn("hour_cos", F.cos(2 * F.lit(PI) * F.col("hour") / 24.0))
        .withColumn("dayofweek", F.dayofweek(F.col("timestamp")))
        .withColumn("day_sin", F.sin(2 * F.lit(PI) * F.col("dayofweek") / 7.0))
        .withColumn("day_cos", F.cos(2 * F.lit(PI) * F.col("dayofweek") / 7.0))
        .withColumn("rolling_mean_cpu_1h", F.avg("cpu_utilization").over(window_1h))
        .withColumn("rolling_std_cpu_1h", F.stddev("cpu_utilization").over(window_1h))
        .withColumn("rolling_mean_cpu_6h", F.avg("cpu_utilization").over(window_6h))
        .withColumn("rolling_mean_net_1h", F.avg("network_in_mbps").over(window_1h))
        .withColumn("rolling_std_net_1h", F.stddev("network_in_mbps").over(window_1h))
        .dropna()
    )

    # 2. VectorAssembler anwenden
    feature_cols = [
        "cpu_utilization",
        "network_in_mbps",
        "hour",
        "hour_sin",
        "hour_cos",
        "dayofweek",
        "day_sin",
        "day_cos",
        "rolling_mean_cpu_1h",
        "rolling_std_cpu_1h",
        "rolling_mean_cpu_6h",
        "rolling_mean_net_1h",
        "rolling_std_net_1h",
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="ml_vector")
    df_vector = assembler.transform(df_features)

    # CRITICAL REFRACTORING: Konvertiere den ML-Vektor in ein primitives Spark Array,
    # damit pyarrow/pandas ihn fehlerfrei lesen können!
    df_assembled = df_vector.withColumn("feature_array", vector_to_array(F.col("ml_vector"))).drop(
        "ml_vector"
    )

    # 3. Chronologischer Split über prozentuale Limits (Vollständig lazy ohne Warnungen!)
    total_rows = df_assembled.count()
    train_limit = int(total_rows * config.data_split["train_ratio"])
    val_limit = train_limit + int(total_rows * config.data_split["val_ratio"])

    # Wir sammeln die genauen Timestamps für die Schnittstellen im sortierten Datenstrom
    timestamps = [
        row["timestamp"] for row in df_assembled.select("timestamp").orderBy("timestamp").collect()
    ]
    train_cut_ts = timestamps[train_limit]
    val_cut_ts = timestamps[val_limit]

    df_train = df_assembled.filter(F.col("timestamp") <= train_cut_ts)
    df_val = df_assembled.filter(
        (F.col("timestamp") > train_cut_ts) & (F.col("timestamp") <= val_cut_ts)
    )
    df_test = df_assembled.filter(F.col("timestamp") > val_cut_ts)

    logger.info(
        f"Exportiere verteilte Splits: Train={df_train.count()} | Val={df_val.count()} | Test={df_test.count()}"
    )

    # 4. Export
    df_train.write.mode("overwrite").parquet(str(processed_dir / "train.parquet"))
    df_val.write.mode("overwrite").parquet(str(processed_dir / "val.parquet"))
    df_test.write.mode("overwrite").parquet(str(processed_dir / "test.parquet"))

    logger.info("✅ Alle Splits erfolgreich als bereinigte Parquet-Strukturen exportiert.")
    spark.stop()


if __name__ == "__main__":
    config = Config()
    run_spark_ingestion(config)
    run_spark_feature_engineering(config)
