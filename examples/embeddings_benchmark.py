#!/usr/bin/env python3
"""
Comprehensive Embedding Model Benchmark for Clustering
======================================================

Scientific benchmarking of embedding models on semantic clustering tasks.
Tests all available embedding models on 50 high-quality sentences across
5 distinct semantic categories to determine which model performs best
for clustering and semantic understanding.

Usage:
    python examples/embeddings_benchmark.py

Categories tested:
1. Scientific Research & Technology (10 sentences)
2. Culinary Arts & Food Culture (10 sentences)
3. Financial Markets & Economics (10 sentences)
4. Environmental Conservation & Nature (10 sentences)
5. Art History & Cultural Heritage (10 sentences)
"""

import time
import random
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import abstractcore
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractcore.embeddings import EmbeddingManager
from abstractcore.embeddings.models import list_available_models, get_model_config


# ============================================================================
# HIGH-QUALITY SENTENCE DATASET (50 sentences, 5 categories, 10 each)
# ============================================================================

SENTENCE_CATEGORIES = {
    "Scientific Research & Technology": [
        "Quantum computers leverage superposition and entanglement to process information exponentially faster than classical systems.",
        "CRISPR-Cas9 gene editing technology enables precise modification of DNA sequences to treat genetic disorders.",
        "Machine learning algorithms trained on massive datasets can identify patterns invisible to human analysis.",
        "The James Webb Space Telescope uses infrared observations to study the formation of early galaxies and exoplanets.",
        "Neuroplasticity research demonstrates the brain's remarkable ability to reorganize neural pathways throughout life.",
        "Renewable energy technologies like perovskite solar cells promise unprecedented efficiency gains in photovoltaic systems.",
        "Synthetic biology engineers microorganisms to produce pharmaceuticals, biofuels, and sustainable materials.",
        "Advanced materials research focuses on developing metamaterials with properties not found in nature.",
        "Precision medicine tailors treatments based on individual genetic profiles and biomarker analysis.",
        "Artificial intelligence systems now surpass human performance in complex strategic games and protein folding prediction."
    ],

    "Culinary Arts & Food Culture": [
        "The Maillard reaction between amino acids and sugars creates complex flavors and aromatic compounds in cooked foods.",
        "French culinary techniques emphasize mother sauces, knife skills, and the importance of mise en place preparation.",
        "Fermentation processes transform raw ingredients into distinctive foods like kimchi, sourdough bread, and aged cheeses.",
        "Japanese kaiseki cuisine presents seasonal ingredients through multiple courses that celebrate natural flavors.",
        "Molecular gastronomy employs scientific principles to create innovative textures and surprising flavor combinations.",
        "Regional wine terroir reflects the unique combination of soil, climate, and traditional winemaking practices.",
        "Street food markets worldwide showcase authentic local flavors and traditional cooking methods passed down generations.",
        "Spice trade routes historically connected distant cultures and transformed global culinary traditions.",
        "Farm-to-table movements emphasize sustainable agriculture and direct relationships between producers and consumers.",
        "Chocolate production from cacao beans involves complex processes of fermentation, roasting, and conching for flavor development."
    ],

    "Financial Markets & Economics": [
        "Central banks use monetary policy tools like interest rates and quantitative easing to manage economic stability.",
        "Cryptocurrency markets operate through blockchain technology and decentralized consensus mechanisms.",
        "Stock market volatility reflects investor sentiment, economic indicators, and geopolitical uncertainty.",
        "Derivatives markets enable risk management through instruments like futures, options, and credit default swaps.",
        "International trade agreements shape global supply chains and competitive advantages between nations.",
        "Private equity firms acquire companies to restructure operations and maximize returns for institutional investors.",
        "Inflation targeting requires careful balance between economic growth and price stability objectives.",
        "Financial regulations like Basel III standards ensure banking system resilience during economic crises.",
        "Exchange-traded funds democratize investment access by providing diversified portfolio exposure at low costs.",
        "Behavioral economics studies how psychological biases influence financial decision-making and market dynamics."
    ],

    "Environmental Conservation & Nature": [
        "Biodiversity loss threatens ecosystem stability as species extinction rates accelerate beyond natural baselines.",
        "Reforestation initiatives combat climate change by sequestering atmospheric carbon in growing forest biomass.",
        "Marine protected areas preserve critical habitat for endangered species and maintain ocean ecosystem health.",
        "Sustainable agriculture practices reduce chemical inputs while maintaining crop yields through ecological methods.",
        "Renewable energy transitions require careful integration of intermittent sources with grid stability requirements.",
        "Wildlife corridors connect fragmented habitats to enable species migration and genetic diversity maintenance.",
        "Coral reef bleaching events result from rising ocean temperatures and acidification caused by climate change.",
        "Conservation biology applies population genetics and ecosystem ecology to protect endangered species.",
        "Green infrastructure incorporates natural systems into urban planning for stormwater management and air quality.",
        "Circular economy principles minimize waste through design strategies that prioritize reuse and recycling."
    ],

    "Art History & Cultural Heritage": [
        "Renaissance artists revolutionized painting through linear perspective, sfumato techniques, and anatomical accuracy.",
        "Gothic cathedrals demonstrate medieval engineering prowess through flying buttresses and soaring pointed arches.",
        "Indigenous art traditions preserve cultural knowledge and spiritual beliefs through symbolic representations and storytelling.",
        "Museum conservation employs scientific analysis to preserve artworks for future generations while maintaining authenticity.",
        "Abstract expressionism emerged as American artists developed new visual languages for emotional and spiritual expression.",
        "Archaeological discoveries provide insights into ancient civilizations through material culture and artistic production.",
        "Baroque architecture features dramatic lighting effects, ornate decoration, and dynamic spatial compositions.",
        "Contemporary art challenges traditional boundaries through multimedia installations and conceptual frameworks.",
        "Cultural heritage preservation balances tourism development with protection of historical sites and traditions.",
        "Digital humanities projects use technology to analyze patterns in artistic production and cultural transmission."
    ]
}

# Flatten into single list with labels for ground truth
SENTENCES = []
GROUND_TRUTH_LABELS = []
CATEGORY_NAMES = list(SENTENCE_CATEGORIES.keys())

for category_id, (category_name, sentences) in enumerate(SENTENCE_CATEGORIES.items()):
    for sentence in sentences:
        SENTENCES.append(sentence)
        GROUND_TRUTH_LABELS.append(category_id)

print(f"✅ Created dataset: {len(SENTENCES)} sentences across {len(CATEGORY_NAMES)} categories")


# ============================================================================
# BENCHMARKING METHODOLOGY
# ============================================================================

class EmbeddingBenchmark:
    """Scientific benchmark for embedding model clustering performance."""

    def __init__(self, seed: int = 42):
        """Initialize benchmark with reproducible randomization."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Create randomized sentence order (same for all models)
        self.randomized_indices = list(range(len(SENTENCES)))
        random.shuffle(self.randomized_indices)

        self.randomized_sentences = [SENTENCES[i] for i in self.randomized_indices]
        self.randomized_labels = [GROUND_TRUTH_LABELS[i] for i in self.randomized_indices]

        print(f"🔀 Randomized dataset order with seed {seed}")

    def benchmark_model(self, model_name: str) -> Dict[str, Any]:
        """Benchmark a single embedding model."""
        print(f"\n{'='*60}")
        print(f"🚀 BENCHMARKING MODEL: {model_name.upper()}")
        print(f"{'='*60}")

        try:
            # Initialize embedding model (HuggingFace cached)
            start_time = time.time()
            model_config = get_model_config(model_name)

            # Use HuggingFace models directly (cached in ~/.cache/huggingface/)
            # Enable trust_remote_code for Nomic models that require custom code
            trust_code = model_name.startswith('nomic-embed')
            embedder = EmbeddingManager(model=model_name, trust_remote_code=trust_code)
            init_time = time.time() - start_time

            print(f"📊 Model Info:")
            print(f"   • Model ID: {model_config.model_id}")
            print(f"   • Dimensions: {model_config.dimension}")
            print(f"   • Size: {model_config.size_mb}MB")
            print(f"   • Multilingual: {model_config.multilingual}")
            print(f"   • Max Length: {model_config.max_sequence_length}")
            print(f"   • Init Time: {init_time:.2f}s")

            # Test embedding generation speed
            print(f"\n🔄 Testing embedding generation for {len(self.randomized_sentences)} sentences...")
            embed_start = time.time()
            embedder.embed_batch(self.randomized_sentences)  # Test speed only
            embed_time = time.time() - embed_start

            print(f"   • Embedding Time: {embed_time:.2f}s ({embed_time/len(self.randomized_sentences)*1000:.1f}ms per sentence)")

            # Test clustering at multiple thresholds
            clustering_results = {}
            thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]

            print(f"\n🔍 Testing clustering at {len(thresholds_to_test)} similarity thresholds...")

            for threshold in thresholds_to_test:
                cluster_start = time.time()
                clusters = embedder.find_similar_clusters(
                    self.randomized_sentences,
                    threshold=threshold,
                    min_cluster_size=2
                )
                cluster_time = time.time() - cluster_start

                # Calculate clustering metrics
                metrics = self._calculate_clustering_metrics(clusters, threshold)
                metrics['cluster_time'] = cluster_time
                clustering_results[threshold] = metrics

                print(f"   Threshold {threshold}: {metrics['num_clusters']} clusters, "
                      f"{metrics['clustered_sentences']}/{len(self.randomized_sentences)} sentences, "
                      f"purity={metrics['purity']:.3f}")

            # Find best threshold based on purity
            best_threshold = max(clustering_results.keys(),
                               key=lambda t: clustering_results[t]['purity'])
            best_result = clustering_results[best_threshold]

            return {
                'model_name': model_name,
                'model_config': model_config,
                'init_time': init_time,
                'embed_time': embed_time,
                'embed_rate': len(self.randomized_sentences) / embed_time,
                'clustering_results': clustering_results,
                'best_threshold': best_threshold,
                'best_result': best_result,
                'status': 'success'
            }

        except Exception as e:
            print(f"❌ ERROR: {e}")
            return {
                'model_name': model_name,
                'status': 'error',
                'error': str(e)
            }

    def _calculate_clustering_metrics(self, clusters: List[List[int]], threshold: float) -> Dict[str, Any]:
        """Calculate clustering quality metrics."""
        if not clusters:
            return {
                'num_clusters': 0,
                'clustered_sentences': 0,
                'purity': 0.0,
                'coverage': 0.0,
                'average_cluster_size': 0.0,
                'category_distribution': {},
                'threshold': threshold
            }

        clustered_indices = set()
        category_purities = []
        cluster_sizes = []
        category_distribution = {cat: 0 for cat in CATEGORY_NAMES}

        for cluster in clusters:
            cluster_sizes.append(len(cluster))
            clustered_indices.update(cluster)

            # Calculate purity for this cluster
            cluster_labels = [self.randomized_labels[idx] for idx in cluster]
            label_counts = {}
            for label in cluster_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                category_distribution[CATEGORY_NAMES[label]] += 1

            # Purity = fraction of most common label
            most_common_count = max(label_counts.values())
            purity = most_common_count / len(cluster)
            category_purities.append(purity)

        # Overall metrics
        overall_purity = np.mean(category_purities) if category_purities else 0.0
        coverage = len(clustered_indices) / len(self.randomized_sentences)
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        return {
            'num_clusters': len(clusters),
            'clustered_sentences': len(clustered_indices),
            'purity': overall_purity,
            'coverage': coverage,
            'average_cluster_size': avg_cluster_size,
            'category_distribution': category_distribution,
            'threshold': threshold,
            'cluster_sizes': cluster_sizes
        }

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on all available models."""
        available_models = list_available_models()

        print(f"🎯 COMPREHENSIVE EMBEDDING MODEL BENCHMARK")
        print(f"{'='*60}")
        print(f"📋 Testing {len(available_models)} models on {len(SENTENCES)} sentences")
        print(f"📂 Categories: {', '.join(CATEGORY_NAMES)}")
        print(f"🔬 Methodology: Clustering at multiple similarity thresholds")
        print(f"📊 Metrics: Purity, Coverage, Speed, Model Efficiency")

        results = {}
        successful_models = []

        for model_name in available_models:
            result = self.benchmark_model(model_name)
            results[model_name] = result

            if result['status'] == 'success':
                successful_models.append(model_name)

        # Generate comparative analysis
        if successful_models:
            analysis = self._generate_comparative_analysis(results, successful_models)
            results['_analysis'] = analysis

        return results

    def _generate_comparative_analysis(self, results: Dict, successful_models: List[str]) -> Dict[str, Any]:
        """Generate comparative analysis of successful models."""
        print(f"\n{'='*60}")
        print(f"📈 COMPARATIVE ANALYSIS")
        print(f"{'='*60}")

        # Extract metrics for comparison
        comparison_data = []
        for model in successful_models:
            result = results[model]
            best = result['best_result']
            config = result['model_config']

            comparison_data.append({
                'model': model,
                'purity': best['purity'],
                'coverage': best['coverage'],
                'num_clusters': best['num_clusters'],
                'embed_rate': result['embed_rate'],
                'dimensions': config.dimension,
                'size_mb': config.size_mb,
                'multilingual': config.multilingual,
                'best_threshold': result['best_threshold']
            })

        # Sort by purity (primary metric)
        comparison_data.sort(key=lambda x: x['purity'], reverse=True)

        print(f"\n🏆 MODEL RANKING BY CLUSTERING PURITY:")
        print(f"{'Rank':<4} {'Model':<15} {'Purity':<7} {'Coverage':<9} {'Clusters':<8} {'Speed':<12} {'Size':<8}")
        print(f"{'-'*70}")

        for rank, data in enumerate(comparison_data, 1):
            print(f"{rank:<4} {data['model']:<15} {data['purity']:.3f}   "
                  f"{data['coverage']:.2f}      {data['num_clusters']:<8} "
                  f"{data['embed_rate']:.1f} s/sec    {data['size_mb']}MB")

        # Find best models by different criteria
        best_purity = max(comparison_data, key=lambda x: x['purity'])
        best_speed = max(comparison_data, key=lambda x: x['embed_rate'])
        best_efficiency = min(comparison_data, key=lambda x: x['size_mb'])

        print(f"\n🎖️  CATEGORY WINNERS:")
        print(f"   🥇 Best Clustering Quality: {best_purity['model']} (purity: {best_purity['purity']:.3f})")
        print(f"   ⚡ Fastest Processing: {best_speed['model']} ({best_speed['embed_rate']:.1f} sentences/sec)")
        print(f"   💾 Most Efficient: {best_efficiency['model']} ({best_efficiency['size_mb']}MB)")

        # Detailed analysis of top 3 models
        print(f"\n🔍 DETAILED ANALYSIS OF TOP 3 MODELS:")
        for rank, data in enumerate(comparison_data[:3], 1):
            model = data['model']
            result = results[model]
            print(f"\n{rank}. {model.upper()}")
            print(f"   • Clustering Purity: {data['purity']:.3f}")
            print(f"   • Coverage: {data['coverage']:.2f} ({int(data['coverage']*50)}/50 sentences clustered)")
            print(f"   • Optimal Threshold: {data['best_threshold']}")
            print(f"   • Processing Speed: {data['embed_rate']:.1f} sentences/sec")
            print(f"   • Model Efficiency: {data['dimensions']} dims, {data['size_mb']}MB")
            print(f"   • Multilingual: {'Yes' if data['multilingual'] else 'No'}")

        # Generate recommendation
        recommendation = self._generate_recommendation(comparison_data)

        return {
            'ranking': comparison_data,
            'winners': {
                'best_purity': best_purity,
                'best_speed': best_speed,
                'best_efficiency': best_efficiency
            },
            'recommendation': recommendation,
            'total_models_tested': len(successful_models),
            'benchmark_methodology': {
                'sentence_count': len(SENTENCES),
                'categories': len(CATEGORY_NAMES),
                'similarity_thresholds': [0.3, 0.4, 0.5, 0.6, 0.7],
                'primary_metric': 'clustering_purity',
                'randomization_seed': self.seed
            }
        }

    def _generate_recommendation(self, comparison_data: List[Dict]) -> Dict[str, str]:
        """Generate usage recommendations based on results."""
        if not comparison_data:
            return {"general": "No successful models to recommend"}

        best_overall = comparison_data[0]  # Sorted by purity
        best_speed = max(comparison_data, key=lambda x: x['embed_rate'])
        best_small = min([d for d in comparison_data if d['purity'] > 0.7],
                        key=lambda x: x['size_mb'], default=comparison_data[0])

        recommendations = {
            "general": f"For best clustering quality, use {best_overall['model']} "
                      f"(purity: {best_overall['purity']:.3f}, {best_overall['size_mb']}MB)",

            "speed_focused": f"For fastest processing, use {best_speed['model']} "
                           f"({best_speed['embed_rate']:.1f} sentences/sec, "
                           f"purity: {best_speed['purity']:.3f})",

            "resource_constrained": f"For resource-constrained environments, use {best_small['model']} "
                                   f"({best_small['size_mb']}MB, purity: {best_small['purity']:.3f})",
        }

        # Add multilingual recommendation if applicable
        multilingual_models = [d for d in comparison_data if d['multilingual']]
        if multilingual_models:
            best_multilingual = multilingual_models[0]  # First in purity ranking
            recommendations["multilingual"] = f"For multilingual support, use {best_multilingual['model']} " \
                                            f"(purity: {best_multilingual['purity']:.3f})"

        return recommendations


def main():
    """Run the comprehensive benchmark."""
    print("🎯 Starting Comprehensive Embedding Model Benchmark")
    print("=" * 60)

    benchmark = EmbeddingBenchmark(seed=42)
    results = benchmark.run_comprehensive_benchmark()

    # Print final summary
    if '_analysis' in results:
        analysis = results['_analysis']
        print(f"\n{'='*60}")
        print(f"🎉 FINAL RECOMMENDATIONS")
        print(f"{'='*60}")

        for use_case, rec in analysis['recommendation'].items():
            print(f"📋 {use_case.replace('_', ' ').title()}: {rec}")

        print(f"\n🔬 BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"   • Models tested: {analysis['total_models_tested']}")
        print(f"   • Best model: {analysis['ranking'][0]['model']} "
              f"(purity: {analysis['ranking'][0]['purity']:.3f})")
        print(f"   • Dataset: {len(SENTENCES)} sentences across {len(CATEGORY_NAMES)} categories")

    return results


if __name__ == "__main__":
    results = main()