        # Create cache directory (thread-safe for parallel executions)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Cache directory created/verified: {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise CompressionCacheError(f"Cannot create cache directory: {e}")

