"""
Compression analytics system for tracking and analyzing compression performance.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from ..utils.structured_logging import get_logger


@dataclass
class CompressionMetrics:
    """Metrics for a single compression operation."""
    timestamp: float
    provider: str
    model: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    processing_time: float
    images_created: int
    method: str  # "glyph", "hybrid", etc.
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "quality_score": self.quality_score,
            "processing_time": self.processing_time,
            "images_created": self.images_created,
            "method": self.method,
            "success": self.success,
            "metadata": self.metadata
        }


class CompressionAnalytics:
    """
    Analytics system for tracking compression performance.

    Collects metrics, analyzes trends, and provides insights for optimization.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize analytics system.

        Args:
            storage_path: Path to store analytics data
        """
        self.logger = get_logger(self.__class__.__name__)
        self.metrics: List[CompressionMetrics] = []
        self.storage_path = storage_path or Path.home() / ".abstractcore" / "analytics" / "compression.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing metrics
        self._load_metrics()

    def _load_metrics(self):
        """Load existing metrics from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.metrics.append(CompressionMetrics(**item))
                self.logger.debug(f"Loaded {len(self.metrics)} historical metrics")
            except Exception as e:
                self.logger.warning(f"Failed to load metrics: {e}")

    def _save_metrics(self):
        """Save metrics to storage."""
        try:
            data = [m.to_dict() for m in self.metrics]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved {len(self.metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def record_compression(
        self,
        provider: str,
        model: str,
        original_tokens: int,
        compressed_tokens: int,
        quality_score: float,
        processing_time: float,
        images_created: int = 0,
        method: str = "glyph",
        success: bool = True,
        metadata: Dict[str, Any] = None
    ) -> CompressionMetrics:
        """
        Record a compression operation.

        Args:
            provider: Provider used
            model: Model used
            original_tokens: Original token count
            compressed_tokens: Compressed token count
            quality_score: Quality score (0-1)
            processing_time: Processing time in seconds
            images_created: Number of images created
            method: Compression method used
            success: Whether compression succeeded
            metadata: Additional metadata

        Returns:
            Created CompressionMetrics object
        """
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0

        metric = CompressionMetrics(
            timestamp=time.time(),
            provider=provider,
            model=model,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            processing_time=processing_time,
            images_created=images_created,
            method=method,
            success=success,
            metadata=metadata or {}
        )

        self.metrics.append(metric)
        self._save_metrics()

        self.logger.info(
            f"Recorded compression: {provider}/{model}, "
            f"{compression_ratio:.1f}x, {quality_score:.2%} quality"
        )

        return metric

    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """
        Get statistics for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Statistics dictionary
        """
        provider_metrics = [m for m in self.metrics if m.provider == provider and m.success]

        if not provider_metrics:
            return {"provider": provider, "no_data": True}

        ratios = [m.compression_ratio for m in provider_metrics]
        qualities = [m.quality_score for m in provider_metrics]
        times = [m.processing_time for m in provider_metrics]

        return {
            "provider": provider,
            "total_compressions": len(provider_metrics),
            "avg_compression_ratio": statistics.mean(ratios),
            "median_compression_ratio": statistics.median(ratios),
            "best_compression_ratio": max(ratios),
            "avg_quality_score": statistics.mean(qualities),
            "avg_processing_time": statistics.mean(times),
            "success_rate": len(provider_metrics) / len([m for m in self.metrics if m.provider == provider])
        }

    def get_model_stats(self, provider: str, model: str) -> Dict[str, Any]:
        """
        Get statistics for a specific model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Statistics dictionary
        """
        model_metrics = [
            m for m in self.metrics
            if m.provider == provider and m.model == model and m.success
        ]

        if not model_metrics:
            return {"provider": provider, "model": model, "no_data": True}

        ratios = [m.compression_ratio for m in model_metrics]
        qualities = [m.quality_score for m in model_metrics]
        times = [m.processing_time for m in model_metrics]
        images = [m.images_created for m in model_metrics]

        return {
            "provider": provider,
            "model": model,
            "total_compressions": len(model_metrics),
            "avg_compression_ratio": statistics.mean(ratios),
            "std_compression_ratio": statistics.stdev(ratios) if len(ratios) > 1 else 0,
            "avg_quality_score": statistics.mean(qualities),
            "avg_processing_time": statistics.mean(times),
            "avg_images_created": statistics.mean(images),
            "percentiles": {
                "p25": statistics.quantiles(ratios, n=4)[0] if len(ratios) > 1 else ratios[0],
                "p50": statistics.median(ratios),
                "p75": statistics.quantiles(ratios, n=4)[2] if len(ratios) > 1 else ratios[0],
            } if ratios else {}
        }

    def get_method_comparison(self) -> Dict[str, Any]:
        """
        Compare different compression methods.

        Returns:
            Comparison statistics
        """
        methods = {}

        for method in set(m.method for m in self.metrics):
            method_metrics = [m for m in self.metrics if m.method == method and m.success]

            if method_metrics:
                ratios = [m.compression_ratio for m in method_metrics]
                qualities = [m.quality_score for m in method_metrics]
                times = [m.processing_time for m in method_metrics]

                methods[method] = {
                    "count": len(method_metrics),
                    "avg_compression": statistics.mean(ratios),
                    "avg_quality": statistics.mean(qualities),
                    "avg_time": statistics.mean(times),
                    "efficiency": statistics.mean(ratios) / statistics.mean(times) if times else 0
                }

        return methods

    def get_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get compression trends over time.

        Args:
            hours: Number of hours to analyze

        Returns:
            Trend analysis
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time and m.success]

        if not recent_metrics:
            return {"no_recent_data": True}

        # Sort by timestamp
        recent_metrics.sort(key=lambda m: m.timestamp)

        # Calculate rolling averages
        window_size = max(1, len(recent_metrics) // 10)
        rolling_ratios = []
        rolling_qualities = []

        for i in range(len(recent_metrics) - window_size + 1):
            window = recent_metrics[i:i + window_size]
            rolling_ratios.append(statistics.mean([m.compression_ratio for m in window]))
            rolling_qualities.append(statistics.mean([m.quality_score for m in window]))

        # Detect trends
        if len(rolling_ratios) > 1:
            ratio_trend = "improving" if rolling_ratios[-1] > rolling_ratios[0] else "declining"
            quality_trend = "improving" if rolling_qualities[-1] > rolling_qualities[0] else "declining"
        else:
            ratio_trend = "stable"
            quality_trend = "stable"

        return {
            "period_hours": hours,
            "total_compressions": len(recent_metrics),
            "ratio_trend": ratio_trend,
            "quality_trend": quality_trend,
            "current_avg_ratio": statistics.mean([m.compression_ratio for m in recent_metrics[-window_size:]]) if recent_metrics else 0,
            "current_avg_quality": statistics.mean([m.quality_score for m in recent_metrics[-window_size:]]) if recent_metrics else 0
        }

    def get_optimization_suggestions(self) -> List[str]:
        """
        Generate optimization suggestions based on analytics.

        Returns:
            List of suggestions
        """
        suggestions = []

        # Analyze recent performance
        recent_metrics = [m for m in self.metrics[-100:] if m.success]  # Last 100 compressions

        if recent_metrics:
            avg_ratio = statistics.mean([m.compression_ratio for m in recent_metrics])
            avg_quality = statistics.mean([m.quality_score for m in recent_metrics])

            # Compression ratio suggestions
            if avg_ratio < 3.0:
                suggestions.append("Consider using more aggressive compression settings (smaller fonts, more columns)")
            elif avg_ratio > 5.0 and avg_quality < 0.90:
                suggestions.append("High compression may be affecting quality - consider balancing settings")

            # Quality suggestions
            if avg_quality < 0.90:
                suggestions.append("Quality scores are low - increase DPI or font size")
            elif avg_quality > 0.95 and avg_ratio < 4.0:
                suggestions.append("Quality is very high - room for more aggressive compression")

            # Provider-specific suggestions
            provider_stats = {}
            for provider in set(m.provider for m in recent_metrics):
                stats = self.get_provider_stats(provider)
                if not stats.get("no_data"):
                    provider_stats[provider] = stats

            if provider_stats:
                best_provider = max(provider_stats.items(), key=lambda x: x[1].get("avg_compression_ratio", 0))
                if best_provider[1]["avg_compression_ratio"] > avg_ratio * 1.2:
                    suggestions.append(f"Provider '{best_provider[0]}' shows better compression - consider using it more")

            # Method suggestions
            method_comparison = self.get_method_comparison()
            if "hybrid" in method_comparison and "glyph" in method_comparison:
                if method_comparison["hybrid"]["avg_compression"] > method_comparison["glyph"]["avg_compression"] * 2:
                    suggestions.append("Hybrid compression shows significant improvement - use it for large documents")

        if not suggestions:
            suggestions.append("Performance is optimal - no specific improvements recommended")

        return suggestions

    def generate_report(self) -> str:
        """
        Generate a comprehensive analytics report.

        Returns:
            Formatted report string
        """
        report = ["=" * 80]
        report.append("COMPRESSION ANALYTICS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total compressions: {len(self.metrics)}")

        # Overall statistics
        if self.metrics:
            successful = [m for m in self.metrics if m.success]
            if successful:
                report.append(f"\nOVERALL STATISTICS:")
                report.append(f"  Success rate: {len(successful)/len(self.metrics):.1%}")
                report.append(f"  Avg compression: {statistics.mean([m.compression_ratio for m in successful]):.1f}x")
                report.append(f"  Avg quality: {statistics.mean([m.quality_score for m in successful]):.1%}")
                report.append(f"  Avg time: {statistics.mean([m.processing_time for m in successful]):.2f}s")

        # Provider breakdown
        providers = set(m.provider for m in self.metrics)
        if providers:
            report.append(f"\nPROVIDER PERFORMANCE:")
            for provider in sorted(providers):
                stats = self.get_provider_stats(provider)
                if not stats.get("no_data"):
                    report.append(f"  {provider}:")
                    report.append(f"    Compressions: {stats['total_compressions']}")
                    report.append(f"    Avg ratio: {stats['avg_compression_ratio']:.1f}x")
                    report.append(f"    Avg quality: {stats['avg_quality_score']:.1%}")

        # Method comparison
        methods = self.get_method_comparison()
        if methods:
            report.append(f"\nMETHOD COMPARISON:")
            for method, stats in methods.items():
                report.append(f"  {method}:")
                report.append(f"    Avg compression: {stats['avg_compression']:.1f}x")
                report.append(f"    Avg quality: {stats['avg_quality']:.1%}")
                report.append(f"    Efficiency: {stats['efficiency']:.2f}")

        # Trends
        trends = self.get_trends(24)
        if not trends.get("no_recent_data"):
            report.append(f"\nRECENT TRENDS (24h):")
            report.append(f"  Compression trend: {trends['ratio_trend']}")
            report.append(f"  Quality trend: {trends['quality_trend']}")
            report.append(f"  Current avg ratio: {trends['current_avg_ratio']:.1f}x")

        # Suggestions
        suggestions = self.get_optimization_suggestions()
        if suggestions:
            report.append(f"\nOPTIMIZATION SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions, 1):
                report.append(f"  {i}. {suggestion}")

        report.append("=" * 80)

        return "\n".join(report)


# Global analytics instance
_analytics_instance = None


def get_analytics() -> CompressionAnalytics:
    """Get global analytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = CompressionAnalytics()
    return _analytics_instance