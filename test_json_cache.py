#!/usr/bin/env python3
"""
Simple test script to verify JSON caching functionality
"""

import sys
import os
sys.path.append('.')

from utils import JSONCache
import tempfile

def test_json_cache():
    """Test basic JSONCache functionality"""
    cache = JSONCache()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        test_data = {'test': 'data', 'number': 42, 'nested': {'key': 'value'}}
        
        result = cache.save_json_cached(temp_file, test_data)
        print(f'Save result: {result}')
        assert result == True, "Save should succeed"
        
        loaded_data = cache.load_json_cached(temp_file)
        print(f'Loaded data: {loaded_data}')
        assert loaded_data == test_data, "Loaded data should match saved data"
        
        loaded_again = cache.load_json_cached(temp_file)
        print(f'Cache hit: {loaded_again == loaded_data}')
        assert loaded_again == loaded_data, "Cache hit should return same data"
        
        default_data = cache.load_json_cached('/non/existent/file.json', {'default': True})
        print(f'Default data: {default_data}')
        assert default_data == {'default': True}, "Should return default for non-existent file"
        
        print('✅ All JSONCache tests passed!')
        return True
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == '__main__':
    success = test_json_cache()
    sys.exit(0 if success else 1)
