#!/usr/bin/env python3
"""
Simple benchmark script to demonstrate JSON caching performance improvement
"""

import os
import json
import time
import tempfile
import threading
from typing import Dict, Any, Optional

class JSONCache:
    """Standalone JSONCache for benchmarking"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(JSONCache, cls).__new__(cls)
                    cls._instance._cache = {}
                    cls._instance._mod_times = {}
                    cls._instance._cache_lock = threading.Lock()
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
            self._mod_times = {}
            self._cache_lock = threading.Lock()
    
    def load_json_cached(self, file_path: str, default: Any = None) -> Any:
        abs_path = os.path.abspath(file_path)
        with self._cache_lock:
            try:
                if not os.path.exists(abs_path):
                    return default
                current_mod_time = os.path.getmtime(abs_path)
                cached_mod_time = self._mod_times.get(abs_path)
                if abs_path in self._cache and cached_mod_time == current_mod_time:
                    return self._cache[abs_path]
                with open(abs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache[abs_path] = data
                    self._mod_times[abs_path] = current_mod_time
                    return data
            except Exception:
                return default
    
    def save_json_cached(self, file_path: str, data: Any, indent: int = 4) -> bool:
        abs_path = os.path.abspath(file_path)
        with self._cache_lock:
            try:
                with open(abs_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=indent, ensure_ascii=False)
                self._cache[abs_path] = data
                self._mod_times[abs_path] = os.path.getmtime(abs_path)
                return True
            except Exception:
                return False

def benchmark_without_cache(file_path, data, iterations=100):
    """Benchmark JSON operations without caching"""
    start_time = time.time()
    
    for _ in range(iterations):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
    
    end_time = time.time()
    return end_time - start_time

def benchmark_with_cache(file_path, data, iterations=100):
    """Benchmark JSON operations with caching"""
    cache = JSONCache()
    start_time = time.time()
    
    for _ in range(iterations):
        cache.save_json_cached(file_path, data)
        
        cache.load_json_cached(file_path)
    
    end_time = time.time()
    return end_time - start_time

def main():
    test_data = {
        'settings': {
            'appearance': {
                'ai_name_color': '#A6E22E',
                'instruction_name_color': '#FFD700',
                'cost_color': '#00FFFF'
            },
            'chromadb': {
                'embedding_model': 'text-embedding-3-small',
                'auto_add_files': True,
                'max_file_size_mb': 5,
                'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__'],
                'file_types': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java']
            }
        },
        'favorites': [
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'provider': 'openai'},
            {'id': 'claude-3-5-sonnet', 'name': 'Claude 3.5 Sonnet', 'provider': 'anthropic'}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        print("JSON Caching Performance Benchmark")
        print("=" * 40)
        
        iterations = 50
        print(f"Running {iterations} iterations of read/write operations...")
        
        time_without_cache = benchmark_without_cache(temp_file, test_data, iterations)
        print(f"Without cache: {time_without_cache:.4f} seconds")
        
        time_with_cache = benchmark_with_cache(temp_file, test_data, iterations)
        print(f"With cache:    {time_with_cache:.4f} seconds")
        
        improvement = ((time_without_cache - time_with_cache) / time_without_cache) * 100
        speedup = time_without_cache / time_with_cache if time_with_cache > 0 else float('inf')
        
        print(f"\nPerformance Improvement:")
        print(f"- Time saved: {improvement:.1f}%")
        print(f"- Speedup: {speedup:.1f}x faster")
        
        print(f"\nNote: Cache provides biggest benefits when the same files")
        print(f"are accessed multiple times, which is common in the AI Chat Terminal")
        print(f"during settings management, favorites handling, and model configuration.")
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == '__main__':
    main()
