#!/usr/bin/env python3
"""
Glyph Compression Test: Privacy Paper Analysis

This script tests AbstractCore's Glyph compression by analyzing a research paper
with and without compression, measuring performance and response quality.

Requirements:
- LMStudio running with qwen3-next-80b model
- PDF file: /Users/albou/projects/preserving_privacy.pdf
"""

import time
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the abstractcore directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abstractcore import create_llm
from abstractcore.compression import GlyphConfig, CompressionOrchestrator
from abstractcore.utils.token_utils import TokenUtils


def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_test_questions():
    """Define the test questions for analysis."""
    return [
        {
            "id": "novelty_and_ideas",
            "question": """What is the novelty and key ideas presented in this article? Please provide a comprehensive analysis covering:
1. The main research problem being addressed
2. Novel contributions and innovations
3. Key methodological approaches
4. Theoretical frameworks or models introduced
5. Practical implications and applications

Please be thorough and specific in your analysis."""
        },
        {
            "id": "figures_and_tables",
            "question": """What is highlighted by the figures and tables in this article? Please analyze:
1. What each figure/table demonstrates or proves
2. Key data points, trends, or patterns shown
3. How the visual elements support the main arguments
4. Any surprising or counterintuitive findings
5. The relationship between different figures/tables

Please reference specific figures and tables by number when possible."""
        }
    ]


def measure_performance(func):
    """Decorator to measure function execution time and memory usage."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        performance_metrics = {
            'execution_time_seconds': round(end_time - start_time, 2),
            'memory_delta_mb': round(end_memory - start_memory, 2),
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat()
        }
        
        return result, performance_metrics
    return wrapper


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0  # Return 0 if psutil not available


def analyze_pdf_content(pdf_path):
    """Analyze the PDF content to get baseline metrics."""
    print(f"📄 Analyzing PDF content: {pdf_path}")
    
    try:
        # Try to extract text for token estimation
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
        
        # Estimate tokens
        estimated_tokens = TokenUtils.estimate_tokens(text_content)
        
        return {
            'file_size_mb': round(Path(pdf_path).stat().st_size / (1024 * 1024), 2),
            'num_pages': len(pdf_reader.pages),
            'estimated_tokens': estimated_tokens,
            'text_length': len(text_content),
            'has_text_content': len(text_content.strip()) > 100
        }
    except Exception as e:
        print(f"⚠️  Could not analyze PDF content: {e}")
        return {
            'file_size_mb': round(Path(pdf_path).stat().st_size / (1024 * 1024), 2),
            'num_pages': 'unknown',
            'estimated_tokens': 'unknown',
            'text_length': 'unknown',
            'has_text_content': 'unknown',
            'analysis_error': str(e)
        }


@measure_performance
def run_test_without_glyph(pdf_path, questions):
    """Run the test without Glyph compression."""
    print("🔄 Running test WITHOUT Glyph compression...")
    
    # Create LLM without Glyph compression
    llm = create_llm(
        "lmstudio",
        model="qwen/qwen3-next-80b",
        base_url="http://localhost:1234/v1"
    )
    
    results = []
    
    for question_data in questions:
        print(f"   Processing question: {question_data['id']}")
        
        question_start = time.time()
        
        try:
            response = llm.generate(
                question_data['question'],
                media=[pdf_path],
                glyph_compression="never"  # Explicitly disable compression
            )
            
            question_time = time.time() - question_start
            
            result = {
                'question_id': question_data['id'],
                'question': question_data['question'],
                'response': response.content,
                'success': True,
                'processing_time_seconds': round(question_time, 2),
                'token_usage': response.usage if response and hasattr(response, 'usage') else None,
                'metadata': response.metadata if response and hasattr(response, 'metadata') and response.metadata else {},
                'compression_used': response.metadata.get('compression_used', False) if response and hasattr(response, 'metadata') and response.metadata else False
            }
            
        except Exception as e:
            result = {
                'question_id': question_data['id'],
                'question': question_data['question'],
                'response': None,
                'success': False,
                'error': str(e),
                'processing_time_seconds': round(time.time() - question_start, 2)
            }
        
        results.append(result)
    
    return results


@measure_performance
def run_test_with_glyph(pdf_path, questions):
    """Run the test with Glyph compression."""
    print("🎨 Running test WITH Glyph compression...")
    
    # Create Glyph configuration optimized for LMStudio
    glyph_config = GlyphConfig(
        enabled=True,
        global_default="always",  # Force compression for testing
        quality_threshold=0.90,   # Slightly lower threshold for local models
        target_compression_ratio=3.0,
        provider_optimization=True
    )
    
    # Customize for LMStudio
    glyph_config.provider_profiles["lmstudio"] = {
        "dpi": 96,
        "font_size": 10,
        "quality_threshold": 0.90,
        "auto_crop_width": True,
        "auto_crop_last_page": True
    }
    
    # Create LLM with Glyph compression
    llm = create_llm(
        "lmstudio",
        model="qwen/qwen3-next-80b",
        base_url="http://localhost:1234/v1",
        glyph_config=glyph_config
    )
    
    results = []
    
    for question_data in questions:
        print(f"   Processing question: {question_data['id']}")
        
        question_start = time.time()
        
        try:
            response = llm.generate(
                question_data['question'],
                media=[pdf_path],
                glyph_compression="always"  # Force compression
            )
            
            question_time = time.time() - question_start
            
            result = {
                'question_id': question_data['id'],
                'question': question_data['question'],
                'response': response.content,
                'success': True,
                'processing_time_seconds': round(question_time, 2),
                'token_usage': response.usage if response and hasattr(response, 'usage') else None,
                'metadata': response.metadata if response and hasattr(response, 'metadata') and response.metadata else {},
                'compression_used': response.metadata.get('compression_used', False) if response and hasattr(response, 'metadata') and response.metadata else False,
                'compression_stats': response.metadata.get('compression_stats') if response and hasattr(response, 'metadata') and response.metadata else None
            }
            
        except Exception as e:
            result = {
                'question_id': question_data['id'],
                'question': question_data['question'],
                'response': None,
                'success': False,
                'error': str(e),
                'processing_time_seconds': round(time.time() - question_start, 2)
            }
        
        results.append(result)
    
    return results


def save_results(results, performance_metrics, test_type, timestamp, pdf_analysis):
    """Save test results to markdown file."""
    filename = f"tests/glyph/{timestamp}-{test_type}.md"
    
    with open(filename, 'w') as f:
        f.write(f"# Glyph Compression Test Results - {test_type.upper()}\n\n")
        f.write(f"**Timestamp**: {timestamp}\n")
        f.write(f"**Test Type**: {test_type}\n")
        f.write(f"**Model**: LMStudio qwen/qwen3-next-80b\n")
        f.write(f"**PDF**: /Users/albou/projects/preserving_privacy.pdf\n\n")
        
        # PDF Analysis
        f.write("## PDF Analysis\n\n")
        for key, value in pdf_analysis.items():
            f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
        f.write("\n")
        
        # Performance Metrics
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Total Execution Time**: {performance_metrics['execution_time_seconds']} seconds\n")
        f.write(f"- **Memory Delta**: {performance_metrics['memory_delta_mb']} MB\n")
        f.write(f"- **Start Time**: {performance_metrics['start_time']}\n")
        f.write(f"- **End Time**: {performance_metrics['end_time']}\n\n")
        
        # Test Results
        for i, result in enumerate(results, 1):
            f.write(f"## Question {i}: {result['question_id'].replace('_', ' ').title()}\n\n")
            
            f.write("### Question\n")
            f.write(f"{result['question']}\n\n")
            
            if result['success']:
                f.write("### Response\n")
                f.write(f"{result['response']}\n\n")
                
                f.write("### Metrics\n")
                f.write(f"- **Processing Time**: {result['processing_time_seconds']} seconds\n")
                f.write(f"- **Success**: {result['success']}\n")
                f.write(f"- **Compression Used**: {result.get('compression_used', 'Unknown')}\n")
                
                if result.get('token_usage'):
                    f.write(f"- **Token Usage**: {result['token_usage']}\n")
                
                if result.get('compression_stats'):
                    f.write("- **Compression Stats**:\n")
                    for key, value in result['compression_stats'].items():
                        f.write(f"  - {key}: {value}\n")
                
                if result.get('metadata'):
                    f.write(f"- **Metadata**: {json.dumps(result['metadata'], indent=2)}\n")
            else:
                f.write("### Error\n")
                f.write(f"**Error**: {result['error']}\n")
                f.write(f"**Processing Time**: {result['processing_time_seconds']} seconds\n")
            
            f.write("\n---\n\n")
    
    print(f"✅ Results saved to: {filename}")
    return filename


def compare_results(no_glyph_results, glyph_results, no_glyph_perf, glyph_perf):
    """Compare results between Glyph and non-Glyph tests."""
    print("\n📊 Comparing Results...")
    
    comparison = {
        'performance_comparison': {
            'no_glyph_time': no_glyph_perf['execution_time_seconds'],
            'glyph_time': glyph_perf['execution_time_seconds'],
            'time_difference': glyph_perf['execution_time_seconds'] - no_glyph_perf['execution_time_seconds'],
            'speedup_factor': round(no_glyph_perf['execution_time_seconds'] / glyph_perf['execution_time_seconds'], 2) if glyph_perf['execution_time_seconds'] > 0 else 'N/A',
            'memory_difference': glyph_perf['memory_delta_mb'] - no_glyph_perf['memory_delta_mb']
        },
        'response_comparison': []
    }
    
    for i, (no_glyph, glyph) in enumerate(zip(no_glyph_results, glyph_results)):
        if no_glyph['success'] and glyph['success']:
            response_comp = {
                'question_id': no_glyph['question_id'],
                'no_glyph_length': len(no_glyph['response']),
                'glyph_length': len(glyph['response']),
                'length_difference': len(glyph['response']) - len(no_glyph['response']),
                'no_glyph_time': no_glyph['processing_time_seconds'],
                'glyph_time': glyph['processing_time_seconds'],
                'time_difference': glyph['processing_time_seconds'] - no_glyph['processing_time_seconds']
            }
        else:
            response_comp = {
                'question_id': no_glyph['question_id'],
                'no_glyph_success': no_glyph['success'],
                'glyph_success': glyph['success'],
                'comparison': 'One or both tests failed'
            }
        
        comparison['response_comparison'].append(response_comp)
    
    return comparison


def main():
    """Main test execution function."""
    print("🧪 Starting Glyph Compression Test")
    print("=" * 50)
    
    # Configuration
    pdf_path = "/Users/albou/projects/preserving_privacy.pdf"
    timestamp = get_timestamp()
    
    # Verify PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Analyze PDF
    pdf_analysis = analyze_pdf_content(pdf_path)
    print(f"📊 PDF Analysis: {pdf_analysis}")
    
    # Create test questions
    questions = create_test_questions()
    print(f"❓ Created {len(questions)} test questions")
    
    try:
        # Test without Glyph compression
        print("\n" + "=" * 50)
        no_glyph_results, no_glyph_perf = run_test_without_glyph(pdf_path, questions)
        no_glyph_file = save_results(no_glyph_results, no_glyph_perf, "no-glyph", timestamp, pdf_analysis)
        
        # Test with Glyph compression
        print("\n" + "=" * 50)
        glyph_results, glyph_perf = run_test_with_glyph(pdf_path, questions)
        glyph_file = save_results(glyph_results, glyph_perf, "glyph", timestamp, pdf_analysis)
        
        # Compare results
        print("\n" + "=" * 50)
        comparison = compare_results(no_glyph_results, glyph_results, no_glyph_perf, glyph_perf)
        
        # Save comparison
        comparison_file = f"tests/glyph/{timestamp}-comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✅ Comparison saved to: {comparison_file}")
        
        # Print summary
        print("\n🎯 Test Summary:")
        print(f"   No-Glyph Time: {no_glyph_perf['execution_time_seconds']}s")
        print(f"   Glyph Time: {glyph_perf['execution_time_seconds']}s")
        print(f"   Speedup Factor: {comparison['performance_comparison']['speedup_factor']}")
        print(f"   Results saved to: tests/glyph/{timestamp}-*.md")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
