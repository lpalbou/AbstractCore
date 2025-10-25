# Phase 3: System Enhancement & Advanced Features

**Status**: Future Development (Post Phases 1-2)
**Timeline**: 1-2 weeks
**Goal**: Add advanced features for production-scale usage

## Strategic Context

After Phase 1 (core integration) and Phase 2 (documentation accuracy), the media system will be **functionally complete** for core use cases. Phase 3 focuses on **production-scale enhancements** and **advanced capabilities**.

### Target Users
- **Production applications** processing hundreds of files daily
- **Enterprise integrations** requiring robust error handling and monitoring
- **Advanced developers** needing fine-grained control and optimization

## Implementation Strategy

### Core Principle: **Performance & Robustness**

Focus on **scaling characteristics** rather than new features:
- **Batch processing** for efficiency
- **Intelligent caching** for performance
- **Advanced monitoring** for observability
- **Graceful degradation** for reliability

### Architecture Decision: **Optional Enhancement Layer**

```
Core System (Phases 1-2) → Enhancement Layer (Phase 3)
        ↑                           ↑
   (Essential)               (Performance/Convenience)
```

All Phase 3 features are **optional enhancements** - core system remains fully functional without them.

## Detailed Implementation Plan

### 1. Audio/Video Processing Support (Priority 1)

**Strategic Decision**: Audio/Video are **fundamentally different** from images/documents:
- **Size**: Much larger files (GB vs MB)
- **Processing**: Requires specialized libraries (FFmpeg, OpenCV)
- **API Support**: Limited provider support for raw audio/video

**Implementation Approach**: **Transcript-First Strategy**

```python
class AudioProcessor(BaseProcessor):
    """Audio processing with transcript extraction"""

    def process_audio(self, file_path: str) -> MediaContent:
        # Strategy 1: Extract transcript using Whisper
        transcript = self._extract_transcript(file_path)

        # Strategy 2: Extract metadata (duration, format, etc.)
        metadata = self._extract_audio_metadata(file_path)

        return MediaContent(
            media_type=MediaType.AUDIO,
            content=transcript,  # Text representation
            content_format=ContentFormat.TEXT,
            metadata={
                'duration': metadata.duration,
                'format': metadata.format,
                'transcript_confidence': transcript.confidence,
                'original_file_path': file_path
            }
        )

class VideoProcessor(BaseProcessor):
    """Video processing with frame sampling and transcript"""

    def process_video(self, file_path: str) -> MediaContent:
        # Strategy 1: Extract key frames as images
        key_frames = self._extract_key_frames(file_path, max_frames=5)

        # Strategy 2: Extract audio transcript
        transcript = self._extract_audio_transcript(file_path)

        # Strategy 3: Combine into structured content
        content = self._combine_frames_and_transcript(key_frames, transcript)

        return MediaContent(
            media_type=MediaType.VIDEO,
            content=content,  # Structured text + image references
            content_format=ContentFormat.STRUCTURED,
            metadata={
                'duration': self._get_duration(file_path),
                'frame_count': len(key_frames),
                'transcript_available': bool(transcript)
            }
        )
```

**Justification**:
- **Practical**: Most LLMs can't process raw audio/video anyway
- **Efficient**: Text transcripts are much smaller and faster to process
- **Compatible**: Works with existing text-based LLM APIs
- **Extensible**: Can add raw audio/video support later if providers support it

**Dependencies**:
```python
# Optional dependencies (not required for core functionality)
AUDIO_VIDEO_DEPS = [
    "openai-whisper>=20231106",  # For transcript extraction
    "opencv-python>=4.8.0",     # For video frame extraction
    "librosa>=0.10.0",          # For audio analysis
]
```

### 2. Intelligent Caching System (Priority 2)

**Problem**: Reprocessing same files repeatedly is inefficient

**Solution**: **Content-Aware Caching**

```python
class MediaCache:
    """Intelligent caching for processed media content"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.hash_strategy = "content+metadata"  # Hash file content + processing settings

    def get_cache_key(self, file_path: str, processor_config: Dict) -> str:
        """Generate cache key based on content and processing configuration"""
        # Hash file content (not just path - handles file updates)
        content_hash = self._hash_file_content(file_path)

        # Hash processing configuration
        config_hash = hashlib.md5(json.dumps(processor_config, sort_keys=True).encode()).hexdigest()

        return f"{content_hash}_{config_hash}"

    def get_cached_result(self, cache_key: str) -> Optional[MediaContent]:
        """Retrieve cached processing result"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return MediaContent.from_dict(json.loads(cache_file.read_text()))
        return None

    def cache_result(self, cache_key: str, result: MediaContent) -> None:
        """Cache processing result"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(result.to_dict(), indent=2))
```

**Integration with AutoMediaHandler**:

```python
class AutoMediaHandler:
    def __init__(self, enable_cache: bool = True, cache_dir: Optional[str] = None):
        self.cache = MediaCache(cache_dir) if enable_cache else None

    def process_file(self, file_path: str) -> ProcessingResult:
        if self.cache:
            cache_key = self.cache.get_cache_key(file_path, self._get_processing_config())
            cached_result = self.cache.get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached result for {file_path}")
                return ProcessingResult(success=True, media_content=cached_result)

        # Process normally
        result = self._process_file_uncached(file_path)

        if self.cache and result.success:
            self.cache.cache_result(cache_key, result.media_content)

        return result
```

**Justification**:
- **Performance**: Avoid reprocessing large PDFs, videos
- **Intelligent**: Content-based hashing detects file changes
- **Optional**: Can be disabled for development/testing
- **Space-efficient**: JSON storage is compact for text content

### 3. Batch Processing System (Priority 3)

**Problem**: Processing many files individually is inefficient

**Solution**: **Parallel Batch Processing**

```python
class BatchMediaProcessor:
    """Efficient batch processing of multiple media files"""

    def __init__(self, max_workers: Optional[int] = None, enable_cache: bool = True):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.handler = AutoMediaHandler(enable_cache=enable_cache)

    def process_batch(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable] = None
    ) -> BatchProcessingResult:
        """Process multiple files in parallel"""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.handler.process_file, path): path
                for path in file_paths
            }

            results = {}
            completed = 0

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                except Exception as e:
                    results[path] = ProcessingResult(
                        success=False,
                        error_message=str(e),
                        file_path=path
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(file_paths))

        return BatchProcessingResult(
            results=results,
            total_files=len(file_paths),
            successful_files=len([r for r in results.values() if r.success]),
            processing_time=time.time() - start_time
        )

    def process_directory(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = True
    ) -> BatchProcessingResult:
        """Process all files in a directory matching pattern"""
        from pathlib import Path

        path = Path(directory)
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        # Filter to supported file types
        supported_files = [
            str(f) for f in files
            if self._is_supported_file_type(f.suffix.lower())
        ]

        return self.process_batch(supported_files)
```

**Justification**:
- **Parallel**: Uses ThreadPoolExecutor for I/O-bound processing
- **Progress tracking**: Supports progress callbacks for UI integration
- **Directory support**: Convenient bulk processing
- **Error isolation**: One file failure doesn't stop batch

### 4. Advanced Monitoring & Observability (Priority 4)

**Problem**: Production usage needs monitoring and debugging capabilities

**Solution**: **Comprehensive Telemetry System**

```python
@dataclass
class MediaProcessingMetrics:
    """Metrics for media processing operations"""
    file_path: str
    file_size: int
    media_type: MediaType
    processor_used: str
    processing_time: float
    cache_hit: bool
    success: bool
    error_type: Optional[str] = None
    memory_usage_mb: Optional[float] = None

class MediaTelemetry:
    """Telemetry collection for media processing"""

    def __init__(self, enable_metrics: bool = True):
        self.enable_metrics = enable_metrics
        self.metrics: List[MediaProcessingMetrics] = []
        self.start_time = time.time()

    def record_processing(self, metrics: MediaProcessingMetrics) -> None:
        """Record processing metrics"""
        if self.enable_metrics:
            self.metrics.append(metrics)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.metrics:
            return {}

        total_files = len(self.metrics)
        successful_files = len([m for m in self.metrics if m.success])
        total_size = sum(m.file_size for m in self.metrics)
        total_time = sum(m.processing_time for m in self.metrics)
        cache_hits = len([m for m in self.metrics if m.cache_hit])

        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'success_rate': successful_files / total_files if total_files > 0 else 0,
            'total_size_mb': total_size / (1024 * 1024),
            'total_processing_time': total_time,
            'average_time_per_file': total_time / total_files if total_files > 0 else 0,
            'cache_hit_rate': cache_hits / total_files if total_files > 0 else 0,
            'throughput_mb_per_second': (total_size / (1024 * 1024)) / total_time if total_time > 0 else 0
        }
```

**Integration with Structured Logging**:

```python
class AutoMediaHandler:
    def __init__(self, telemetry: Optional[MediaTelemetry] = None):
        self.telemetry = telemetry or MediaTelemetry()

    def process_file(self, file_path: str) -> ProcessingResult:
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            result = self._process_file_internal(file_path)

            # Record metrics
            metrics = MediaProcessingMetrics(
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                media_type=result.media_content.media_type,
                processor_used=self._get_processor_name(),
                processing_time=time.time() - start_time,
                cache_hit=getattr(result, 'cache_hit', False),
                success=result.success,
                memory_usage_mb=(self._get_memory_usage() - start_memory) / (1024 * 1024)
            )

            self.telemetry.record_processing(metrics)
            return result

        except Exception as e:
            # Record error metrics
            metrics = MediaProcessingMetrics(
                file_path=file_path,
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                media_type=MediaType.UNKNOWN,
                processor_used="none",
                processing_time=time.time() - start_time,
                cache_hit=False,
                success=False,
                error_type=type(e).__name__
            )

            self.telemetry.record_processing(metrics)
            raise
```

**Justification**:
- **Observability**: Track performance and reliability in production
- **Debugging**: Detailed error tracking and performance analysis
- **Optimization**: Identify bottlenecks and improvement opportunities
- **Optional**: Can be disabled for simple use cases

### 5. Streaming Support for Large Files (Priority 5)

**Problem**: Large files (100MB+ PDFs, videos) can cause memory issues

**Solution**: **Chunked Processing with Streaming**

```python
class StreamingProcessor:
    """Process large files in chunks to manage memory usage"""

    def __init__(self, chunk_size: int = 8 * 1024 * 1024):  # 8MB chunks
        self.chunk_size = chunk_size

    def process_large_pdf(self, file_path: str) -> Iterator[MediaContent]:
        """Process PDF in page chunks"""
        import pymupdf4llm

        doc = pymupdf4llm.Document(file_path)
        total_pages = len(doc)

        # Process in chunks of 10 pages
        chunk_size = 10
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)

            chunk_content = pymupdf4llm.to_markdown(
                doc=doc,
                pages=list(range(start_page, end_page))
            )

            yield MediaContent(
                media_type=MediaType.DOCUMENT,
                content=chunk_content,
                content_format=ContentFormat.TEXT,
                metadata={
                    'chunk_index': start_page // chunk_size,
                    'page_range': f"{start_page + 1}-{end_page}",
                    'total_pages': total_pages,
                    'file_path': file_path
                }
            )

    def process_large_video(self, file_path: str) -> Iterator[MediaContent]:
        """Process video in time-based chunks"""
        # Extract frames at regular intervals
        duration = self._get_video_duration(file_path)
        chunk_duration = 60  # 1 minute chunks

        for start_time in range(0, int(duration), chunk_duration):
            end_time = min(start_time + chunk_duration, duration)

            # Extract representative frame from this time chunk
            frame = self._extract_frame_at_time(file_path, start_time + chunk_duration // 2)

            # Extract audio transcript for this chunk
            transcript = self._extract_transcript_chunk(file_path, start_time, end_time)

            yield MediaContent(
                media_type=MediaType.VIDEO,
                content=f"Frame: {frame}\nTranscript: {transcript}",
                content_format=ContentFormat.STRUCTURED,
                metadata={
                    'time_range': f"{start_time}s-{end_time}s",
                    'total_duration': duration,
                    'chunk_index': start_time // chunk_duration
                }
            )
```

**Integration with LLM Calls**:

```python
def process_large_file_with_llm(file_path: str, prompt: str, llm) -> str:
    """Process large file by streaming chunks to LLM"""
    processor = StreamingProcessor()

    if file_path.endswith('.pdf'):
        chunks = processor.process_large_pdf(file_path)
    elif file_path.endswith(('.mp4', '.avi', '.mov')):
        chunks = processor.process_large_video(file_path)
    else:
        # Fallback to normal processing
        return llm.generate(prompt, media=[file_path]).content

    # Process each chunk and combine results
    chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"{prompt}\n\nProcessing chunk {i + 1} of document:\n{chunk.content}"
        result = llm.generate(chunk_prompt)
        chunk_results.append(result.content)

    # Combine chunk results
    final_prompt = f"Combine these analysis results into a coherent summary:\n\n" + "\n\n".join(chunk_results)
    return llm.generate(final_prompt).content
```

**Justification**:
- **Memory efficiency**: Process large files without loading entirely into memory
- **Scalability**: Handle files of any size
- **Quality**: Maintain analysis quality through chunking strategy
- **Flexibility**: Different chunking strategies for different media types

## Configuration and Settings

### 1. Enhanced Configuration System

```python
@dataclass
class MediaConfig:
    """Configuration for media processing system"""

    # Core settings
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB

    # Performance settings
    batch_processing_workers: int = 4
    enable_streaming: bool = True
    streaming_chunk_size: int = 8 * 1024 * 1024  # 8MB

    # Feature toggles
    enable_audio_video: bool = False  # Requires additional dependencies
    enable_telemetry: bool = True
    enable_advanced_caching: bool = True

    # Provider-specific settings
    openai_image_detail: str = "high"  # "low", "high", or "auto"
    anthropic_max_images: int = 20
    local_vision_fallback: bool = True

    @classmethod
    def from_env(cls) -> 'MediaConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_cache=os.getenv('ABSTRACTCORE_MEDIA_CACHE', 'true').lower() == 'true',
            max_file_size=int(os.getenv('ABSTRACTCORE_MEDIA_MAX_SIZE', str(100 * 1024 * 1024))),
            batch_processing_workers=int(os.getenv('ABSTRACTCORE_MEDIA_WORKERS', '4')),
            enable_audio_video=os.getenv('ABSTRACTCORE_MEDIA_AUDIO_VIDEO', 'false').lower() == 'true'
        )
```

### 2. CLI Integration

```bash
# Configuration commands
abstractcore --media-config --enable-cache
abstractcore --media-config --disable-audio-video
abstractcore --media-config --set-max-file-size 200MB
abstractcore --media-config --set-workers 8

# Batch processing commands
abstractcore --process-directory /path/to/media --pattern "*.pdf"
abstractcore --process-batch file1.jpg file2.pdf file3.docx

# Performance analysis
abstractcore --media-stats --show-performance
abstractcore --media-stats --show-cache-stats
```

## Implementation Timeline

### Week 1: Core Enhancements
**Days 1-2**: Audio/Video Processing
- Implement AudioProcessor with Whisper integration
- Implement VideoProcessor with frame extraction
- Add optional dependency management

**Days 3-4**: Caching System
- Implement MediaCache with content-based hashing
- Integrate with AutoMediaHandler
- Add cache management CLI commands

**Day 5**: Batch Processing
- Implement BatchMediaProcessor
- Add ThreadPoolExecutor-based parallel processing
- Create directory processing utilities

### Week 2: Advanced Features
**Days 1-2**: Telemetry and Monitoring
- Implement MediaTelemetry system
- Add performance metrics collection
- Create analysis and reporting tools

**Days 3-4**: Streaming Support
- Implement StreamingProcessor for large files
- Add chunked processing for PDFs and videos
- Create memory-efficient processing pipeline

**Day 5**: Integration and Testing
- Comprehensive testing of all Phase 3 features
- Performance benchmarking
- Documentation updates

## Success Criteria

### 1. Performance Requirements
- ✅ **Batch processing**: 10x faster than individual file processing
- ✅ **Memory efficiency**: Process 1GB+ files without memory issues
- ✅ **Cache effectiveness**: 80%+ cache hit rate for repeated files
- ✅ **Audio/Video support**: Extract meaningful content from multimedia

### 2. Robustness Requirements
- ✅ **Error isolation**: Batch processing handles individual failures gracefully
- ✅ **Monitoring**: Comprehensive metrics for production debugging
- ✅ **Configuration**: Easy customization for different use cases
- ✅ **Scalability**: Linear performance scaling with worker count

### 3. Integration Requirements
- ✅ **Backward compatibility**: All existing functionality unchanged
- ✅ **Optional features**: Core system works without Phase 3 dependencies
- ✅ **CLI integration**: New features accessible via command line
- ✅ **Configuration management**: Environment variable support

## Post-Phase 3 System Characteristics

After Phase 3 completion, the AbstractCore media system will be:

### **Enterprise-Ready**
- **Scalable**: Handle thousands of files efficiently
- **Monitorable**: Comprehensive observability for production
- **Configurable**: Extensive customization options
- **Robust**: Graceful handling of edge cases and failures

### **Developer-Friendly**
- **Flexible**: Multiple processing strategies for different needs
- **Extensible**: Clear patterns for adding new processors
- **Debuggable**: Rich telemetry and error reporting
- **Performant**: Optimized for production workloads

### **Future-Proof**
- **Modular**: New features can be added without breaking existing functionality
- **Standards-based**: Uses established patterns and libraries
- **Configurable**: Easy adaptation to new requirements
- **Maintainable**: Clean architecture supports long-term evolution

This positions AbstractCore's media handling as a **production-grade system** suitable for enterprise applications while maintaining the simplicity and ease of use for basic scenarios.