"""Result caching for inference optimization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("crackgraphai.cache")


class ResultCache:
    """Simple LRU cache for inference results with thread-safe operations."""
    
    def __init__(
        self,
        ttl: int = 3600,  # Time-to-live in seconds
        max_size: int = 1000,  # Max number of cached items
        persistent: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.ttl = ttl
        self.max_size = max_size
        self.persistent = persistent
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()  # Thread-safe lock for concurrent access
        self._last_save_time = time.time()
        self._save_interval = 60  # Save to disk every 60 seconds max
        
        if persistent:
            self.cache_dir.mkdir(exist_ok=True)
            self._load_persistent_cache()
    
    def _load_persistent_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "inference_cache.pkl"
        if cache_file.exists():
            try:
                with self._lock:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        self._cache = data.get('cache', {})
                        self._access_times = data.get('access_times', {})
                    logger.info(f"Loaded {len(self._cache)} items from persistent cache")
            except Exception as e:
                logger.warning(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self):
        """Save cache to disk incrementally."""
        if not self.persistent:
            return
        
        # Throttle saves to avoid excessive disk I/O
        now = time.time()
        if now - self._last_save_time < self._save_interval:
            return
        
        cache_file = self.cache_dir / "inference_cache.pkl"
        try:
            with self._lock:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'cache': self._cache,
                        'access_times': self._access_times,
                    }, f)
                self._last_save_time = now
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._access_times:
            return True
        
        age = time.time() - self._access_times[key]
        return age > self.ttl
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full (thread-safe)."""
        with self._lock:
            if len(self._cache) < self.max_size:
                return
            
            # Remove expired entries first
            expired = [k for k in list(self._cache.keys()) if self._is_expired(k)]
            for k in expired:
                del self._cache[k]
                del self._access_times[k]
            
            # If still full, remove LRU
            if len(self._cache) >= self.max_size:
                sorted_keys = sorted(
                    self._access_times.keys(),
                    key=lambda k: self._access_times[k]
                )
                to_remove = sorted_keys[:len(sorted_keys) // 4]  # Remove 25%
                for k in to_remove:
                    del self._cache[k]
                    del self._access_times[k]
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached result if available and not expired (thread-safe)."""
        with self._lock:
            if key not in self._cache:
                return None
            
            if self._is_expired(key):
                del self._cache[key]
                del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def set(self, key: str, value: Dict):
        """Cache a result (thread-safe)."""
        self._evict_if_needed()
        
        # Convert numpy arrays to lists for serialization
        serializable_value = self._make_serializable(value)
        
        with self._lock:
            self._cache[key] = serializable_value
            self._access_times[key] = time.time()
        
        # Periodic save (outside lock to avoid blocking)
        if self.persistent:
            self._save_persistent_cache()
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable for caching."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj
    
    def clear(self):
        """Clear all cached results (thread-safe)."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
        
        if self.persistent:
            cache_file = self.cache_dir / "inference_cache.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics (thread-safe)."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'persistent': self.persistent,
            }


class DiskCache:
    """Disk-based cache for large results."""
    
    def __init__(
        self,
        cache_dir: str = ".cache/disk",
        ttl: int = 86400,  # 24 hours
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.meta_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.meta_file.exists():
            try:
                with open(self.meta_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.meta_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def _get_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash prefix for directory distribution
        prefix = key[:2]
        dir_path = self.cache_dir / prefix
        dir_path.mkdir(exist_ok=True)
        return dir_path / f"{key}.npy"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached numpy array."""
        if key not in self.metadata:
            return None
        
        # Check TTL
        age = time.time() - self.metadata[key]['timestamp']
        if age > self.ttl:
            self.delete(key)
            return None
        
        path = self._get_path(key)
        if not path.exists():
            del self.metadata[key]
            self._save_metadata()
            return None
        
        try:
            return np.load(path)
        except Exception as e:
            logger.warning(f"Failed to load cached array: {e}")
            return None
    
    def set(self, key: str, array: np.ndarray):
        """Cache numpy array."""
        path = self._get_path(key)
        
        try:
            np.save(path, array)
            self.metadata[key] = {
                'timestamp': time.time(),
                'shape': list(array.shape),
                'dtype': str(array.dtype),
            }
            self._save_metadata()
        except Exception as e:
            logger.warning(f"Failed to cache array: {e}")
    
    def delete(self, key: str):
        """Delete cached item."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
        
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
    
    def clear(self):
        """Clear all cached items."""
        for key in list(self.metadata.keys()):
            self.delete(key)
        
        self.metadata = {}
        self._save_metadata()
