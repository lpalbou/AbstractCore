"""
Compression cache system for Glyph.
"""

import os
import json
import time
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from .exceptions import CompressionCacheError
from ..utils.structured_logging import get_logger


class CompressionCache:
    """Cache system for compressed content."""
    
    def __init__(self, cache_dir: str = None, max_size_gb: float = 1.0, ttl_days: int = 7):
        """
        Initialize compression cache.
        
        Args:
            cache_dir: Cache directory path
            max_size_gb: Maximum cache size in GB
            ttl_days: Time-to-live in days
        """
        self.logger = get_logger(self.__class__.__name__)
        
        if cache_dir is None:
            cache_dir = str(Path.home() / ".abstractcore" / "glyph_cache")
        
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = ttl_days * 24 * 3600
        
        # Create cache directory (thread-safe for parallel executions)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Cache directory created/verified: {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise CompressionCacheError(f"Cannot create cache directory: {e}")

        # Initialize metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        self.logger.debug(f"Initialized compression cache at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            'entries': {},
            'created': time.time(),
            'last_cleanup': time.time()
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, content: str, config: Dict[str, Any]) -> str:
        """Generate cache key from content and configuration."""
        # Create a hash of content + configuration
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:8]
        
        return f"{content_hash}_{config_hash}"
    
    def get(self, cache_key: str) -> Optional[List[Path]]:
        """
        Get cached compression result.
        
        Args:
            cache_key: Cache key
            
        Returns:
            List of image paths if cached, None otherwise
        """
        if cache_key not in self.metadata['entries']:
            return None
        
        entry = self.metadata['entries'][cache_key]
        
        # Check TTL
        if time.time() - entry['created'] > self.ttl_seconds:
            self._remove_entry(cache_key)
            return None
        
        # Check if files still exist
        image_paths = [Path(path) for path in entry['image_paths']]
        if not all(path.exists() for path in image_paths):
            self._remove_entry(cache_key)
            return None
        
        # Update access time
        entry['last_accessed'] = time.time()
        self._save_metadata()
        
        self.logger.debug(f"Cache hit for key {cache_key}")
        return image_paths
    
    def set(self, cache_key: str, image_paths: List[Path], 
            compression_stats: Dict[str, Any] = None):
        """
        Store compression result in cache.
        
        Args:
            cache_key: Cache key
            image_paths: List of rendered image paths
            compression_stats: Optional compression statistics
        """
        try:
            # Create cache entry directory
            entry_dir = self.cache_dir / cache_key
            entry_dir.mkdir(exist_ok=True)
            
            # Copy images to cache
            cached_paths = []
            for i, image_path in enumerate(image_paths):
                if image_path.exists():
                    cached_path = entry_dir / f"image_{i}{image_path.suffix}"
                    shutil.copy2(image_path, cached_path)
                    cached_paths.append(str(cached_path))
            
            # Store metadata
            entry = {
                'created': time.time(),
                'last_accessed': time.time(),
                'image_paths': cached_paths,
                'compression_stats': compression_stats or {},
                'size_bytes': sum(Path(path).stat().st_size for path in cached_paths)
            }
            
            self.metadata['entries'][cache_key] = entry
            self._save_metadata()
            
            # Cleanup if needed
            self._cleanup_if_needed()
            
            self.logger.debug(f"Cached compression result for key {cache_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache compression result: {e}")
            raise CompressionCacheError(f"Cache storage failed: {e}")
    
    def _remove_entry(self, cache_key: str):
        """Remove cache entry."""
        if cache_key in self.metadata['entries']:
            # Remove files
            entry_dir = self.cache_dir / cache_key
            if entry_dir.exists():
                shutil.rmtree(entry_dir, ignore_errors=True)
            
            # Remove metadata
            del self.metadata['entries'][cache_key]
            self._save_metadata()
    
    def _cleanup_if_needed(self):
        """Cleanup cache if size or TTL limits exceeded."""
        now = time.time()
        
        # Skip if cleaned up recently
        if now - self.metadata.get('last_cleanup', 0) < 3600:  # 1 hour
            return
        
        self.logger.debug("Starting cache cleanup")
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.metadata['entries'].items():
            if now - entry['created'] > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        # Check size limit
        total_size = sum(entry.get('size_bytes', 0) for entry in self.metadata['entries'].values())
        
        if total_size > self.max_size_bytes:
            # Remove least recently accessed entries
            entries_by_access = sorted(
                self.metadata['entries'].items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )
            
            while total_size > self.max_size_bytes and entries_by_access:
                key, entry = entries_by_access.pop(0)
                total_size -= entry.get('size_bytes', 0)
                self._remove_entry(key)
        
        self.metadata['last_cleanup'] = now
        self._save_metadata()
        
        self.logger.debug(f"Cache cleanup completed. Entries: {len(self.metadata['entries'])}")
    
    def clear(self):
        """Clear all cache entries."""
        try:
            # Remove all entry directories
            for entry_dir in self.cache_dir.iterdir():
                if entry_dir.is_dir() and entry_dir.name != "cache_metadata.json":
                    shutil.rmtree(entry_dir, ignore_errors=True)
            
            # Reset metadata
            self.metadata = {
                'entries': {},
                'created': time.time(),
                'last_cleanup': time.time()
            }
            self._save_metadata()
            
            self.logger.info("Cache cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise CompressionCacheError(f"Cache clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        entries = self.metadata['entries']
        total_size = sum(entry.get('size_bytes', 0) for entry in entries.values())
        
        return {
            'total_entries': len(entries),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
            'cache_dir': str(self.cache_dir),
            'ttl_days': self.ttl_seconds / (24 * 3600),
            'oldest_entry': min((entry['created'] for entry in entries.values()), default=time.time()),
            'newest_entry': max((entry['created'] for entry in entries.values()), default=time.time())
        }

