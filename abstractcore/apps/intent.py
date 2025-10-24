#!/usr/bin/env python3
"""
AbstractCore Intent Analyzer CLI Application

Usage:
    python -m abstractcore.apps.intent <file_path_or_text> [options]

Options:
    --context <context>             Context type (standalone, conversational, document, interactive, default: standalone)
    --depth <depth>                 Analysis depth (surface, underlying, comprehensive, default: underlying)
    --focus <focus>                 Specific focus area for intent analysis (e.g., "business motivations", "emotional drivers")
    --format <format>               Output format (json, yaml, plain, default: json)
    --output <output>               Output file path (optional, prints to console if not provided)
    --chunk-size <size>             Chunk size in characters (default: 8000, max: 32000)
    --provider <provider>           LLM provider (requires --model)
    --model <model>                 LLM model (requires --provider)
    --max-tokens <tokens>           Maximum total tokens for LLM context (default: 32000)
    --max-output-tokens <tokens>    Maximum tokens for LLM output generation (default: 8000)
    --conversation-mode             Analyze as conversation (expects multiple messages)
    --focus-participant <role>      In conversation mode, focus on specific participant (user, assistant, etc.)
    --verbose                       Show detailed progress information
    --timeout <seconds>             HTTP timeout for LLM providers (default: 300)
    --help                          Show this help message

Note: Deception analysis based on psychological markers is always included in intent analysis.

Examples:
    # Single text analysis
    python -m abstractcore.apps.intent "I was wondering if you could help me understand this concept?"
    python -m abstractcore.apps.intent document.txt --depth comprehensive --verbose
    
    # Conversation analysis
    python -m abstractcore.apps.intent conversation.txt --conversation-mode --focus-participant user
    
    # Advanced options
    python -m abstractcore.apps.intent email.txt --context document --focus "business objectives" --output analysis.json
    python -m abstractcore.apps.intent chat.txt --context conversational --depth surface --format plain
    python -m abstractcore.apps.intent query.txt --provider openai --model gpt-4o-mini --depth comprehensive
    
    # Comprehensive psychological analysis (includes deception assessment)
    python -m abstractcore.apps.intent suspicious_message.txt --depth comprehensive
"""

import argparse
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from ..processing import BasicIntentAnalyzer, IntentContext, IntentDepth
from ..core.factory import create_llm


def get_app_defaults(app_name: str) -> tuple[str, str]:
    """Get default provider and model for an app."""
    try:
        from ..config import get_config_manager
        config_manager = get_config_manager()
        return config_manager.get_app_default(app_name)
    except Exception:
        # Fallback to hardcoded defaults if config unavailable
        hardcoded_defaults = {
            'summarizer': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'extractor': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'judge': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'intent': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'cli': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
        }
        return hardcoded_defaults.get(app_name, ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'))


def read_session_file(file_path: str) -> list[dict]:
    """
    Read and parse a BasicSession JSON file
    
    Args:
        file_path: Path to the session JSON file
        
    Returns:
        List of message dictionaries
    """
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both new archive format and legacy format
    if "schema_version" in data and "messages" in data:
        # New archive format
        messages_data = data["messages"]
    else:
        # Legacy format or direct messages
        messages_data = data.get("messages", [])
    
    # Convert to simple format expected by intent analyzer
    messages = []
    for msg_data in messages_data:
        messages.append({
            "role": msg_data["role"],
            "content": msg_data["content"]
        })
    
    return messages


def read_file_content(file_path: str) -> str:
    """
    Read content from various file types

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be read
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path_obj.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Try to read as text file
    try:
        # Try UTF-8 first
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to other encodings
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path_obj, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all text encodings fail, try binary read and decode
        try:
            with open(file_path_obj, 'rb') as f:
                content = f.read()
                # Try to decode as text
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            raise Exception(f"Cannot read file {file_path}: {e}")


def parse_conversation_text(text: str) -> list[dict]:
    """
    Parse conversation text into message format
    
    Expected formats:
    - "USER: message\nASSISTANT: response\n..."
    - "user: message\nassistant: response\n..."
    - "[USER]: message\n[ASSISTANT]: response\n..."
    """
    messages = []
    lines = text.strip().split('\n')
    
    current_role = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for role indicators
        role_found = None
        content_start = None
        
        # Try different formats
        for prefix in ['USER:', 'ASSISTANT:', 'SYSTEM:', '[USER]:', '[ASSISTANT]:', '[SYSTEM]:', 'user:', 'assistant:', 'system:']:
            if line.upper().startswith(prefix.upper()):
                role_found = prefix.replace(':', '').replace('[', '').replace(']', '').lower()
                content_start = line[len(prefix):].strip()
                break
        
        if role_found:
            # Save previous message if exists
            if current_role and current_content:
                messages.append({
                    'role': current_role,
                    'content': '\n'.join(current_content).strip()
                })
            
            # Start new message
            current_role = role_found
            current_content = [content_start] if content_start else []
        else:
            # Continue current message
            if current_role:
                current_content.append(line)
    
    # Save final message
    if current_role and current_content:
        messages.append({
            'role': current_role,
            'content': '\n'.join(current_content).strip()
        })
    
    return messages


def format_intent_output(result, format_type: str, conversation_mode: bool = False, analysis_time: float = None) -> str:
    """Format intent analysis output for display"""
    
    if format_type == "json":
        if conversation_mode and isinstance(result, dict):
            # Multiple participants
            return json.dumps(result, indent=2, default=str)
        else:
            # Single analysis
            return json.dumps(result.dict() if hasattr(result, 'dict') else result, indent=2, default=str)
    
    elif format_type == "yaml":
        if conversation_mode and isinstance(result, dict):
            # Multiple participants
            return yaml.dump(result, default_flow_style=False, default=str)
        else:
            # Single analysis
            return yaml.dump(result.dict() if hasattr(result, 'dict') else result, default_flow_style=False, default=str)
    
    elif format_type == "plain":
        output_lines = []
        
        if conversation_mode and isinstance(result, dict):
            # Multiple participants
            output_lines.append("🎯 CONVERSATION INTENT ANALYSIS")
            output_lines.append("=" * 50)
            
            for participant, analysis in result.items():
                output_lines.append(f"\n👤 PARTICIPANT: {participant.upper()}")
                output_lines.append("-" * 30)
                output_lines.extend(_format_single_analysis_plain(analysis, analysis_time))
        else:
            # Single analysis
            output_lines.append("🎯 INTENT ANALYSIS")
            output_lines.append("=" * 40)
            output_lines.extend(_format_single_analysis_plain(result, analysis_time))
        
        return "\n".join(output_lines)
    
    else:
        raise ValueError(f"Unknown format: {format_type}")


def _format_single_analysis_plain(analysis, analysis_time: float = None) -> list[str]:
    """Format a single intent analysis in plain text"""
    lines = []
    
    # Primary Intent
    lines.append(f"\n🎯 PRIMARY INTENT: {analysis.primary_intent.intent_type.value.replace('_', ' ').title()}")
    lines.append(f"   Description: {analysis.primary_intent.description}")
    lines.append(f"   Underlying Goal: {analysis.primary_intent.underlying_goal}")
    lines.append(f"   Emotional Undertone: {analysis.primary_intent.emotional_undertone}")
    lines.append(f"   Confidence: {analysis.primary_intent.confidence:.2f}")
    lines.append(f"   Urgency Level: {analysis.primary_intent.urgency_level:.2f}")
    
    # Deception Analysis for Primary Intent (always included)
    if analysis.primary_intent.deception_analysis:
        deception = analysis.primary_intent.deception_analysis
        lines.append(f"\n🔍 DECEPTION ANALYSIS:")
        lines.append(f"   Deception Likelihood: {deception.deception_likelihood:.2f}")
        lines.append(f"   Narrative Consistency: {deception.narrative_consistency:.2f}")
        lines.append(f"   Temporal Coherence: {deception.temporal_coherence:.2f}")
        lines.append(f"   Emotional Congruence: {deception.emotional_congruence:.2f}")
        
        if deception.linguistic_markers:
            lines.append(f"   Linguistic Markers: {', '.join(deception.linguistic_markers)}")
        
        if deception.deception_evidence:
            lines.append(f"   Evidence Indicating Deception:")
            for evidence in deception.deception_evidence:
                lines.append(f"     • {evidence}")
        
        if deception.authenticity_evidence:
            lines.append(f"   Evidence Indicating Authenticity:")
            for evidence in deception.authenticity_evidence:
                lines.append(f"     • {evidence}")
    
    # Secondary Intents
    if analysis.secondary_intents:
        lines.append(f"\n🔄 SECONDARY INTENTS ({len(analysis.secondary_intents)}):")
        for i, intent in enumerate(analysis.secondary_intents, 1):
            lines.append(f"   {i}. {intent.intent_type.value.replace('_', ' ').title()}")
            lines.append(f"      Goal: {intent.underlying_goal}")
            lines.append(f"      Confidence: {intent.confidence:.2f}")
            
            # Deception analysis for secondary intents (always included)
            if intent.deception_analysis:
                deception = intent.deception_analysis
                lines.append(f"      Deception Likelihood: {deception.deception_likelihood:.2f}")
                if deception.linguistic_markers:
                    lines.append(f"      Linguistic Markers: {', '.join(deception.linguistic_markers[:2])}")  # Limit for brevity
    
    # Analysis Metadata
    lines.append(f"\n📊 ANALYSIS METADATA:")
    lines.append(f"   Intent Complexity: {analysis.intent_complexity:.2f}")
    lines.append(f"   Overall Confidence: {analysis.overall_confidence:.2f}")
    lines.append(f"   Words Analyzed: {analysis.word_count_analyzed:,}")
    lines.append(f"   Analysis Depth: {analysis.analysis_depth.value.title()}")
    lines.append(f"   Context Type: {analysis.context_type.value.title()}")
    if analysis_time is not None:
        lines.append(f"   Analysis Time: {analysis_time:.1f}s")
    
    # Contextual Factors
    if analysis.contextual_factors:
        lines.append(f"\n🌍 CONTEXTUAL FACTORS:")
        for factor in analysis.contextual_factors:
            lines.append(f"   • {factor}")
    
    # Response Approach
    lines.append(f"\n💡 SUGGESTED RESPONSE APPROACH:")
    lines.append(f"   {analysis.suggested_response_approach}")
    
    return lines


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AbstractCore Intent Analyzer - Identify and analyze intents behind text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "I was wondering if you could help me understand this concept?"
  %(prog)s document.txt --depth comprehensive --verbose
  %(prog)s conversation.txt --conversation-mode --focus-participant user
  %(prog)s email.txt --context document --focus "business objectives" --output analysis.json
        """
    )
    
    # Required argument
    parser.add_argument(
        'input',
        help='Text to analyze or file path containing text'
    )
    
    # Analysis configuration
    parser.add_argument(
        '--context',
        choices=['standalone', 'conversational', 'document', 'interactive'],
        default='standalone',
        help='Context type for analysis (default: standalone)'
    )
    
    parser.add_argument(
        '--depth',
        choices=['surface', 'underlying', 'comprehensive'],
        default='underlying',
        help='Analysis depth (default: underlying)'
    )
    
    
    parser.add_argument(
        '--focus',
        type=str,
        help='Specific focus area for intent analysis (e.g., "business motivations", "emotional drivers")'
    )
    
    # Conversation mode
    parser.add_argument(
        '--conversation-mode',
        action='store_true',
        help='Analyze as conversation with multiple participants'
    )
    
    parser.add_argument(
        '--focus-participant',
        type=str,
        help='In conversation mode, focus on specific participant (user, assistant, etc.)'
    )
    
    # Output options
    parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'plain'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional, prints to console if not provided)'
    )
    
    # Processing options
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8000,
        help='Chunk size in characters for long documents (default: 8000, max: 32000)'
    )
    
    # LLM configuration
    parser.add_argument(
        '--provider',
        type=str,
        help='LLM provider (requires --model)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='LLM model (requires --provider)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32000,
        help='Maximum total tokens for LLM context (default: 32000)'
    )
    
    parser.add_argument(
        '--max-output-tokens',
        type=int,
        default=8000,
        help='Maximum tokens for LLM output generation (default: 8000)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug information including raw LLM responses and JSON parsing details'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='HTTP timeout for LLM providers in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.chunk_size > 32000:
        print("❌ Error: chunk-size cannot exceed 32000 characters")
        sys.exit(1)
    
    if (args.provider and not args.model) or (args.model and not args.provider):
        print("❌ Error: Both --provider and --model must be specified together")
        sys.exit(1)
    
    # Note: We'll validate focus_participant after we potentially auto-enable conversation_mode for session files
    
    try:
        # Determine if input is a file or direct text
        input_text = ""
        messages_from_session = None
        
        if Path(args.input).exists():
            file_path = Path(args.input)
            if args.verbose:
                print(f"📖 Reading file: {args.input}")
            
            # Check if it's a session JSON file
            if file_path.suffix.lower() == '.json':
                try:
                    # Try to read as session file first
                    messages_from_session = read_session_file(args.input)
                    if args.verbose:
                        print(f"📋 Detected session file with {len(messages_from_session)} messages")
                        print("🔄 Automatically enabling conversation mode")
                    args.conversation_mode = True  # Auto-enable conversation mode for session files
                except (json.JSONDecodeError, KeyError):
                    # If it fails, fall back to reading as text
                    if args.verbose:
                        print("📄 JSON file doesn't appear to be a session file, reading as text")
                    input_text = read_file_content(args.input)
            else:
                # Regular text file
                input_text = read_file_content(args.input)
        else:
            # Treat as direct text input
            input_text = args.input
        
        # Validate we have content to analyze
        if not messages_from_session and not input_text.strip():
            print("❌ Error: No text content to analyze")
            sys.exit(1)
        
        if args.verbose and input_text:
            print(f"📝 Text length: {len(input_text)} characters")
        
        # Now validate focus_participant after potentially auto-enabling conversation_mode
        if args.focus_participant and not args.conversation_mode:
            print("❌ Error: --focus-participant requires --conversation-mode")
            sys.exit(1)
        
        # Get LLM configuration
        if args.provider and args.model:
            provider = args.provider
            model = args.model
        else:
            # Use app defaults
            provider, model = get_app_defaults('intent')
        
        if args.verbose:
            print(f"🤖 Using LLM: {provider}/{model}")
        
        # Create LLM instance
        llm = create_llm(
            provider=provider,
            model=model,
            max_tokens=args.max_tokens,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout
        )
        
        # Create intent analyzer
        analyzer = BasicIntentAnalyzer(
            llm=llm,
            max_chunk_size=args.chunk_size,
            max_tokens=args.max_tokens,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            debug=args.debug
        )
        
        # Convert string enums
        context_type = IntentContext(args.context)
        depth = IntentDepth(args.depth)
        
        # Perform analysis
        start_time = time.time()
        
        if args.conversation_mode:
            if args.verbose:
                print("🗣️  Analyzing conversation intents...")
            
            # Use messages from session file or parse from text
            if messages_from_session:
                messages = messages_from_session
            else:
                # Parse conversation from text
                messages = parse_conversation_text(input_text)
                
                if not messages:
                    print("❌ Error: Could not parse conversation format. Expected format:")
                    print("USER: message\\nASSISTANT: response\\n...")
                    sys.exit(1)
            
            if args.verbose:
                print(f"📋 Parsed {len(messages)} messages")
            
            # Analyze conversation intents (deception analysis always included)
            result = analyzer.analyze_conversation_intents(
                messages=messages,
                focus_participant=args.focus_participant,
                depth=depth
            )
        else:
            if args.verbose:
                print("🎯 Analyzing text intents...")
            
            # Analyze single text (deception analysis always included)
            result = analyzer.analyze_intent(
                text=input_text,
                context_type=context_type,
                depth=depth,
                focus=args.focus
            )
        
        analysis_time = time.time() - start_time
        
        if args.verbose:
            print(f"✅ Analysis completed in {analysis_time:.1f} seconds")
        
        # Format output
        formatted_output = format_intent_output(result, args.format, args.conversation_mode, analysis_time)
        
        # Save or print result
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            
            if args.verbose:
                print(f"💾 Results saved to: {args.output}")
        else:
            print(formatted_output)
    
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during intent analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
