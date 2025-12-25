"""
Monitoring - Métriques Prometheus et Logging Structuré

Ce module fournit:
- Collecteur de métriques Prometheus
- Logger structuré JSON
- Tableaux de bord de monitoring
- Alertes et notifications
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from functools import wraps
from collections import defaultdict
import threading
import os


class MetricsCollector:
    """
    Collecteur de métriques compatible Prometheus.

    Types de métriques:
    - Counter: Compteur monotone
    - Gauge: Valeur instantanée
    - Histogram: Distribution des valeurs
    - Summary: Résumé statistique
    """

    def __init__(self, namespace: str = "cifar10"):
        """
        Initialise le collecteur de métriques.

        Args:
            namespace: Préfixe pour toutes les métriques
        """
        self.namespace = namespace
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

        # Métriques par défaut
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialise les métriques par défaut."""
        self.gauge_set("model_loaded", 0)
        self.gauge_set("api_status", 1)
        self.counter_inc("app_starts_total", 1)

    def counter_inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Incrémente un compteur.

        Args:
            name: Nom de la métrique
            value: Valeur à ajouter
            labels: Labels optionnels
        """
        key = self._build_key(name, labels)
        with self._lock:
            self.counters[key] += value

    def gauge_set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Définit la valeur d'une gauge.

        Args:
            name: Nom de la métrique
            value: Valeur à définir
            labels: Labels optionnels
        """
        key = self._build_key(name, labels)
        with self._lock:
            self.gauges[key] = value

    def gauge_inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Incrémente une gauge."""
        key = self._build_key(name, labels)
        with self._lock:
            self.gauges[key] = self.gauges.get(key, 0) + value

    def gauge_dec(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Décrémente une gauge."""
        self.gauge_inc(name, -value, labels)

    def histogram_observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Ajoute une observation à un histogramme.

        Args:
            name: Nom de la métrique
            value: Valeur observée
            labels: Labels optionnels
        """
        key = self._build_key(name, labels)
        with self._lock:
            self.histograms[key].append(value)
            # Limiter la taille de l'historique
            if len(self.histograms[key]) > 10000:
                self.histograms[key] = self.histograms[key][-5000:]

    def _build_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Construit une clé unique pour la métrique."""
        full_name = f"{self.namespace}_{name}"
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{full_name}{{{label_str}}}"
        return full_name

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne toutes les métriques."""
        with self._lock:
            metrics = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {}
            }

            # Calculer les statistiques des histogrammes
            for name, values in self.histograms.items():
                if values:
                    import numpy as np
                    metrics["histograms"][name] = {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": np.mean(values),
                        "p50": np.percentile(values, 50),
                        "p90": np.percentile(values, 90),
                        "p99": np.percentile(values, 99)
                    }

        return metrics

    def export_prometheus_format(self) -> str:
        """Exporte les métriques au format Prometheus."""
        lines = []

        with self._lock:
            # Counters
            for name, value in self.counters.items():
                lines.append(f"# TYPE {name.split('{')[0]} counter")
                lines.append(f"{name} {value}")

            # Gauges
            for name, value in self.gauges.items():
                lines.append(f"# TYPE {name.split('{')[0]} gauge")
                lines.append(f"{name} {value}")

            # Histograms
            for name, values in self.histograms.items():
                if values:
                    import numpy as np
                    base_name = name.split('{')[0]
                    labels = name[len(base_name):] if '{' in name else ""

                    lines.append(f"# TYPE {base_name} histogram")

                    # Buckets
                    buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                    for bucket in buckets:
                        count = sum(1 for v in values if v <= bucket)
                        bucket_labels = f'le="{bucket}"'
                        if labels:
                            bucket_labels = labels[1:-1] + "," + bucket_labels
                        lines.append(f"{base_name}_bucket{{{bucket_labels}}} {count}")

                    # +Inf bucket
                    inf_labels = 'le="+Inf"'
                    if labels:
                        inf_labels = labels[1:-1] + "," + inf_labels
                    lines.append(f"{base_name}_bucket{{{inf_labels}}} {len(values)}")

                    # Sum et count
                    lines.append(f"{base_name}_sum{labels} {sum(values)}")
                    lines.append(f"{base_name}_count{labels} {len(values)}")

        return "\n".join(lines)

    def timing(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Décorateur pour mesurer le temps d'exécution.

        Usage:
            @metrics.timing("function_duration")
            def my_function():
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    self.histogram_observe(name, duration, labels)
            return wrapper
        return decorator


class StructuredLogger:
    """
    Logger structuré au format JSON pour une meilleure observabilité.

    Fonctionnalités:
    - Logs structurés JSON
    - Contexte automatique (timestamp, level, etc.)
    - Rotation des fichiers
    - Filtrage par niveau
    """

    def __init__(
        self,
        name: str = "cifar10",
        log_dir: str = "logs",
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialise le logger structuré.

        Args:
            name: Nom du logger
            log_dir: Répertoire pour les fichiers de log
            level: Niveau de logging minimum
            console_output: Activer la sortie console
            file_output: Activer la sortie fichier
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.level = level
        self.console_output = console_output
        self.file_output = file_output

        self._context: Dict[str, Any] = {}
        self._setup_logger()

    def _setup_logger(self):
        """Configure le logger."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.handlers = []

        # Formatter JSON
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }

                # Ajouter les extras
                if hasattr(record, 'extra_fields'):
                    log_entry.update(record.extra_fields)

                return json.dumps(log_entry, default=str)

        formatter = JsonFormatter()

        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.file_output:
            log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_context(self, **kwargs):
        """Définit le contexte global pour tous les logs."""
        self._context.update(kwargs)

    def clear_context(self):
        """Efface le contexte global."""
        self._context.clear()

    def _log(self, level: int, message: str, **kwargs):
        """Log avec contexte et extras."""
        extra_fields = {**self._context, **kwargs}

        record = logging.LogRecord(
            name=self.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra_fields

        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log niveau DEBUG."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log niveau INFO."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log niveau WARNING."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log niveau ERROR."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log niveau CRITICAL."""
        self._log(logging.CRITICAL, message, **kwargs)

    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        latency_ms: float,
        **kwargs
    ):
        """Log une prédiction avec ses métriques."""
        self.info(
            "Prediction completed",
            event_type="prediction",
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            **kwargs
        )

    def log_training_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        learning_rate: float,
        **kwargs
    ):
        """Log les métriques d'une epoch d'entraînement."""
        self.info(
            f"Training epoch {epoch} completed",
            event_type="training_epoch",
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            learning_rate=learning_rate,
            **kwargs
        )

    def log_model_loaded(self, model_path: str, model_name: str, **kwargs):
        """Log le chargement d'un modèle."""
        self.info(
            f"Model loaded: {model_name}",
            event_type="model_loaded",
            model_path=model_path,
            model_name=model_name,
            **kwargs
        )

    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        **kwargs
    ):
        """Log une requête API."""
        level = logging.INFO if status_code < 400 else logging.ERROR
        self._log(
            level,
            f"{method} {endpoint} - {status_code}",
            event_type="api_request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            **kwargs
        )


class AlertManager:
    """
    Gestionnaire d'alertes pour le monitoring.

    Fonctionnalités:
    - Définition de règles d'alerte
    - Notification (console, fichier, webhook)
    - Agrégation des alertes
    """

    def __init__(self, log_dir: str = "logs/alerts"):
        """
        Initialise le gestionnaire d'alertes.

        Args:
            log_dir: Répertoire pour les fichiers d'alertes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.rules: List[Dict[str, Any]] = []
        self.active_alerts: List[Dict[str, Any]] = []
        self.logger = StructuredLogger("alerts", str(self.log_dir))

    def add_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: str = "warning",
        message_template: str = ""
    ):
        """
        Ajoute une règle d'alerte.

        Args:
            name: Nom de la règle
            condition: Fonction qui retourne True si l'alerte doit être déclenchée
            severity: Niveau de sévérité (info, warning, critical)
            message_template: Template du message d'alerte
        """
        self.rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "message_template": message_template
        })

    def evaluate(self, metrics: Dict[str, Any]):
        """
        Évalue toutes les règles par rapport aux métriques.

        Args:
            metrics: Dictionnaire des métriques actuelles
        """
        for rule in self.rules:
            try:
                if rule["condition"](metrics):
                    self._trigger_alert(rule, metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule['name']}: {e}")

    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Déclenche une alerte."""
        alert = {
            "name": rule["name"],
            "severity": rule["severity"],
            "message": rule["message_template"].format(**metrics) if rule["message_template"] else rule["name"],
            "timestamp": datetime.now().isoformat(),
            "metrics_snapshot": metrics
        }

        self.active_alerts.append(alert)

        # Logger l'alerte
        if rule["severity"] == "critical":
            self.logger.critical(alert["message"], alert=alert)
        elif rule["severity"] == "warning":
            self.logger.warning(alert["message"], alert=alert)
        else:
            self.logger.info(alert["message"], alert=alert)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne les alertes actives."""
        return self.active_alerts

    def clear_alerts(self):
        """Efface toutes les alertes actives."""
        self.active_alerts.clear()


# Instance globale des métriques
metrics = MetricsCollector()

# Instance globale du logger
logger = StructuredLogger()
