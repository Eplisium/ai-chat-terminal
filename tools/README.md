# ACT Tools System

The ACT Tools System allows you to easily create and manage AI tools that can be used during chat sessions. Tools are automatically discovered and can be enabled/disabled through the GUI.

## Creating a New Tool

To create a new tool:

1. Create a new Python file in the `tools` directory with the naming pattern `*_tool.py` (e.g., `translate_tool.py`)
2. Implement the `execute(arguments)` function with proper docstrings
3. The tool will be automatically discovered and added to the GUI

### Tool Requirements

1. **File Naming**: Tool files must:
   - End with `_tool.py` (e.g., `weather_tool.py`, `translate_tool.py`)
   - Be placed in the `tools` directory
   - Have unique names to avoid conflicts

2. **Function Requirements**:
   - Must have an `execute(arguments)` function
   - Must accept a dictionary of arguments
   - Must return a string result
   - Should include proper error handling
   - Should have descriptive docstrings with parameter descriptions

3. **Security Requirements**:
   - Implement proper access checks for file operations
   - Add logging for important operations
   - Handle errors gracefully and provide informative messages
   - Validate all inputs before processing
   - Use safe defaults for optional parameters

4. **Documentation Requirements**:
   - First line should be a brief description
   - Use `:param name: description` format for parameters
   - Use `:return: description` for return value
   - Document security considerations if applicable
   - Include usage examples in docstring

### Tool Template

```python
import logging

# Set up logging if needed
logger = logging.getLogger(__name__)

def execute(arguments: Dict) -> str:
    """Short description of what your tool does.
    
    :param param1: Description of first parameter
    :param param2: Description of second parameter
    :return: Description of what the tool returns
    
    Example:
        >>> result = execute({
        ...     "param1": "value1",
        ...     "param2": "value2"
        ... })
    """
    try:
        # Get parameters with safe defaults
        param1 = arguments.get('param1', '')
        param2 = arguments.get('param2', '')
        
        # Validate inputs
        if not param1:
            return "Error: param1 is required"
        
        # Your tool implementation here
        result = do_something(param1, param2)
        
        # Log successful operation
        logger.info(f"Operation successful with params: {param1}, {param2}")
        
        return result
    except Exception as e:
        # Log error and return informative message
        logger.error(f"Error in tool execution: {str(e)}")
        return f"Error: {str(e)}"
```

### Example Tool (File Operations)

Here's an example of a file operations tool with proper security and logging:

```python
def execute(arguments: Dict) -> str:
    """Execute file operations tool for browsing and managing files.
    
    :param operation: Operation to perform (info, list, exists, find, size, create, delete, copy, move, write, read, search)
    :param path: Target path for the operation
    :param detailed: Get detailed information (for info operation)
    :param recursive: List subdirectories recursively (for list/find operations)
    :param pattern: Glob pattern for filtering (for list/find operations)
    :param content: Content to write (for write operation)
    :param destination: Destination path (for copy/move operations)
    :param mkdir: Create directory instead of file (for create operation)
    :param start_line: Start line number for read operation (1-based, inclusive)
    :param end_line: End line number for read operation (1-based, inclusive)
    :param query: Search query for searching file contents (for search operation)
    :return: Operation result or error message
    
    Example:
        >>> # List directory contents
        >>> result = execute({
        ...     "operation": "list",
        ...     "path": "C:\\Users",
        ...     "recursive": True
        ... })
        >>> 
        >>> # Search file contents
        >>> result = execute({
        ...     "operation": "search",
        ...     "path": "C:\\Projects",
        ...     "query": "TODO"
        ... })
    """
    try:
        # Implementation with security checks and logging
        pass
    except Exception as e:
        return f"Error: {str(e)}"
```

## Tool Configuration

Tools are configured through:
1. **Discovery**: The system scans for `*_tool.py` files
2. **Documentation**: Tool descriptions and parameters are extracted from docstrings
3. **Configuration**: Settings are stored in `tools_config.json`
4. **GUI Integration**: Tools appear in the AI Tools settings menu

### Configuration Options

Each tool in `tools_config.json` has:
- `enabled`: Whether the tool is available for use
- `description`: Automatically extracted from docstring
- `implementation`: The tool's Python file name
- `parameters`: Schema generated from docstring parameters

### Configuration Stability

The tool configuration system is designed to be stable:
1. Existing tool configurations are preserved between sessions
2. New tools are automatically discovered and added
3. Removed tools are cleaned up from configuration
4. Tool configurations are only updated when explicitly requested

## Using Tools

1. Enable tools in the AI settings menu:
   ```
   AI Settings > AI Tools > Toggle Tools
   ```

2. Configure tool-specific settings if needed:
   ```
   AI Settings > AI Tools > Configure Tool
   ```

3. Use tools through the chat interface by providing the required parameters

## Best Practices

1. **Security**:
   - Always validate user inputs
   - Implement proper access controls
   - Use logging for auditing
   - Handle errors gracefully
   - Use safe defaults

2. **Performance**:
   - Handle large files efficiently
   - Implement pagination where appropriate
   - Use async operations for long-running tasks
   - Cache results when possible

3. **User Experience**:
   - Provide clear error messages
   - Include progress indicators
   - Support cancellation for long operations
   - Maintain backward compatibility

4. **Code Quality**:
   - Write comprehensive docstrings
   - Follow Python style guidelines
   - Add type hints
   - Include usage examples
   - Write unit tests

## Troubleshooting

1. If a tool isn't appearing:
   - Check the file naming (`*_tool.py`)
   - Verify the `execute()` function exists
   - Check the docstring format
   - Look for errors in the logs

2. If a tool isn't working:
   - Check the logs for errors
   - Verify the parameters match the schema
   - Test the tool in isolation
   - Check access permissions

## Dependencies

Common dependencies used by tools:
- `logging`: For operation logging
- `pathlib`: For path manipulation
- `typing`: For type hints
- `python-magic`: For file type detection
- `chardet`: For encoding detection