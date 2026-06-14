from datetime import datetime
from hashlib import sha256
from pydantic import BaseModel, Field


class HubServerSchema(BaseModel):
    """Repräsentiert die Tabellenstruktur des HUB_SERVER im Data Vault 2.0."""

    hk_server: str = Field(description="SHA-256 Hash Key des Business Keys")
    instance_id: str = Field(description="Eindeutige AWS Instance ID (Business Key)")
    load_timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_source: str

    @classmethod
    def generate_hash_key(cls, instance_id: str) -> str:
        """Erzeugt den deterministischen DV 2.0 Hash Key."""
        cleaned_bk = str(instance_id).strip().upper()
        return sha256(cleaned_bk.encode("utf-8")).hexdigest()


class SatServerTelemetrySchema(BaseModel):
    """Repräsentiert die Tabellenstruktur des SAT_SERVER_TELEMETRY."""

    hk_server: str
    timestamp: datetime = Field(description="Der eigentliche Event-Zeitstempel aus dem Log")
    load_timestamp: datetime = Field(default_factory=datetime.utcnow)
    hash_diff: str = Field(description="SHA-256 Hash über die Payload zur Delta-Erkennung")

    # Payload (Nutzdaten)
    cpu_utilization: float
    network_in: float = 0.0
    network_out: float = 0.0

    record_source: str

    @classmethod
    def generate_hash_diff(cls, cpu: float, net_in: float, net_out: float) -> str:
        """Erzeugt ein Hash Diff über alle reinen Nutzdaten-Spalten."""
        concat_str = f"{cpu};{net_in};{net_out}"
        return sha256(concat_str.encode("utf-8")).hexdigest()
