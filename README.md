# ACT (AI Chat Terminal)

A versatile command-line chat interface that supports multiple AI providers including OpenAI and OpenRouter, featuring rich text formatting and comprehensive file handling capabilities.

## Features

### AI Provider Support
- Multiple AI provider integration (OpenAI and OpenRouter)
- Interactive model selection menu with provider-specific features
- Favorite models management for quick access
- Customizable system instructions for AI behavior

### Document Handling
- Support for multiple document formats:
  - Microsoft Word (DOCX)
  - PDF documents
  - OpenDocument Text (ODT)
  - Rich Text Format (RTF)
  - Plain text files with multiple encoding support

### Chat Features
- Interactive chat interface with rich text formatting
- File and directory reference system:
  - View file contents: `[[file:path/to/file]]`
  - List directory contents: `[[dir:path/to/directory]]`
  - View codebase contents: `[[codebase:path/to/codebase]]` (Note: Partial implementation)
- Image embedding support:
  - Local images: `[[img:path/to/image.jpg]]`
  - URL images: `[[img:https://example.com/image.jpg]]`
- Chat session management:
  - Save chat history (`/save`)
  - Clear screen and history (`/clear`)
  - Insert multiline text (`/insert`)
  - Exit options ('exit', 'quit', or Ctrl+C)

### Interface
- Rich console interface with syntax highlighting
- Progress indicators for AI responses
- Comprehensive logging system
- Error handling and reporting

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)
- API keys for OpenAI and/or OpenRouter
- Optional dependencies for document handling:
  - python-docx (DOCX support)
  - PyPDF2 (PDF support)
  - odfpy (ODT support)
  - pyth (RTF support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eplisium/ai-chat-terminal.git
cd ai-chat-terminal
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Usage

Run the script:
```bash
python main.py
```

### Navigation
The interactive menu provides:
1. Provider selection (OpenAI/OpenRouter)
2. Model selection with detailed information
3. Favorites management
4. System instruction customization
5. Chat session initiation

### Chat Commands
During chat sessions:
- `/save` - Save the current chat history (JSON and text formats)
- `/clear` - Clear screen and reset chat history
- `/insert` - Insert multiline text (end with END on new line)
- `exit`, `quit`, or Ctrl+C - End the session

### File References
Reference files in your messages:
```
[[file:example.py]]              # View file contents
[[file:"path with spaces.txt"]]  # Paths with spaces need quotes
[[dir:project/src]]              # List directory contents
[[codebase:src/*.py]]           # View Python files in src (partial implementation)
[[img:image.jpg]]               # Embed local image
[[img:"https://..."]]           # Embed image from URL
```

## License

MIT License - See LICENSE file for details 