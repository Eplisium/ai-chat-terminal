# ACT Tools System

The ACT Tools System allows you to easily create and manage AI tools that can be used during chat sessions. Tools are automatically discovered and can be enabled/disabled through the GUI.

## Creating a New Tool

To create a new tool:

1. Create a new Python file in the `tools` directory with the naming pattern `*_tool.py` (e.g., `translate_tool.py`)
2. Implement the `execute(arguments)` function with proper docstrings
3. The tool will be automatically discovered and added to the GUI

### Tool Template

```python
def execute(arguments: Dict) -> str:
    """Short description of what your tool does.
    
    :param param1: Description of first parameter
    :param param2: Description of second parameter
    :return: Description of what the tool returns
    """
    # Get parameters from arguments dict
    param1 = arguments.get('param1', '')
    param2 = arguments.get('param2', '')
    
    try:
        # Your tool implementation here
        result = do_something(param1, param2)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

### Example Tool

Here's an example of a translation tool:

```python
def execute(arguments: Dict) -> str:
    """Translate text between languages.
    
    :param text: The text to translate
    :param source_lang: Source language code (e.g., 'en', 'es')
    :param target_lang: Target language code (e.g., 'fr', 'de')
    :return: The translated text
    """
    text = arguments.get('text', '')
    source_lang = arguments.get('source_lang', 'auto')
    target_lang = arguments.get('target_lang', 'en')
    
    try:
        # Implementation here
        translated_text = translate_text(text, source_lang, target_lang)
        return f"Translation from {source_lang} to {target_lang}: {translated_text}"
    except Exception as e:
        return f"Error translating text: {str(e)}"
```

## Tool Requirements

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

3. **Docstring Format**:
   - First line should be a brief description
   - Use `:param name: description` format for parameters
   - Use `:return: description` for return value
   - Parameters in docstring will be used to generate the tool's schema

## Tool Configuration

Tools are automatically configured through:
1. Discovery: The system scans for `*_tool.py` files
2. Documentation: Tool descriptions and parameters are extracted from docstrings
3. Configuration: Settings are stored in `tools_config.json`
4. GUI Integration: Tools appear in the AI Tools settings menu

### Configuration Options

Each tool in `tools_config.json` has:
- `enabled`: Whether the tool is available for use
- `description`: Automatically extracted from docstring
- `implementation`: The tool's Python file name
- `parameters`: Schema generated from docstring parameters

## Using Tools

1. Enable tools in the AI settings menu:
   ```
   AI Settings > AI Tools > Toggle Tools
   ```

2. Enable/disable individual tools:
   ```
   AI Settings > AI Tools > [Tool Name]
   ```

3. Tools can be used during chat sessions when:
   - The overall tools feature is enabled
   - The specific tool is enabled
   - The AI model supports function calling

## Best Practices

1. **Error Handling**:
   - Always include try/except blocks
   - Return meaningful error messages
   - Handle missing or invalid parameters gracefully

2. **Documentation**:
   - Write clear, concise descriptions
   - Document all parameters
   - Include example usage if complex

3. **Security**:
   - Validate all inputs
   - Don't expose sensitive information
   - Limit file system access to workspace

4. **Performance**:
   - Keep tools focused and efficient
   - Handle timeouts appropriately
   - Cache results when possible

## Example Tools

The system includes several example tools:
- `calculate_tool.py`: Mathematical calculations
- `time_tool.py`: Time and date information
- `weather_tool.py`: Weather information
- `system_tool.py`: System information
- `fileops_tool.py`: File operations
- `search_tool.py`: Web search functionality

Study these examples to understand best practices for tool implementation. 