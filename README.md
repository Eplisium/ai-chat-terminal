# ACT (AI Chat Terminal)

A versatile command-line chat interface that supports multiple AI providers including OpenAI, Anthropic, and OpenRouter, featuring rich text formatting and comprehensive file handling capabilities.

## Features

### AI Provider Support
- Multiple AI provider integration (OpenAI, Anthropic, and OpenRouter)
- Interactive model selection menu with provider-specific features
- Favorite models management for quick access
- Customizable system instructions for AI behavior
- Support for image-enabled models (where available)

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
  - View codebase contents: `[[codebase:path/to/codebase]]`
- Image embedding support:
  - Local images: `[[img:path/to/image.jpg]]`
  - URL images: `[[img:https://example.com/image.jpg]]`
  - Automatic format detection and conversion
- Chat session management:
  - Save chat history (`/save`)
  - Clear screen and history (`/clear`)
  - Insert multiline text (`/insert`)
  - End chat session (`/end`)
  - Exit options (`exit`, `quit`, `bye`, or Ctrl+C)

### Interface
- Rich console interface with syntax highlighting
- Progress indicators for AI responses
- Comprehensive logging system with rotation
- Detailed error handling and reporting
- Interactive menu system with provider categorization

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)
- API keys for supported providers:
  - OpenAI API key
  - Anthropic API key (optional)
  - OpenRouter API key (optional)
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
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Usage

Run the script:
```bash
python main.py
```

### Navigation
The interactive menu provides:
1. Provider selection (OpenAI/Anthropic/OpenRouter)
2. Favorite models management
3. Model selection with detailed information
4. System instruction customization
5. Chat session initiation

### Chat Commands
During chat sessions:
- `/save` - Save the current chat history (JSON and text formats)
- `/clear` - Clear screen and reset chat history
- `/insert` - Insert multiline text (end with END on new line)
- `/end` - End the chat session
- `exit`, `quit`, `bye`, or Ctrl+C - End the session

### File References
Reference files in your messages:
```
[[file:example.py]]              # View file contents
[[file:"path with spaces.txt"]]  # Paths with spaces need quotes
[[dir:project/src]]              # List directory contents
[[codebase:src/*.py]]           # View Python files in src
[[img:image.jpg]]               # Embed local image
[[img:"https://..."]]           # Embed image from URL
```

### Chat History
Chat histories are automatically saved in both JSON and text formats, organized by:
- Provider (OpenAI, Anthropic, OpenRouter)
- Model company (for OpenRouter)
- Model name
- Timestamp

## License

MIT License - See LICENSE file for details 