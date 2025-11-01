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
