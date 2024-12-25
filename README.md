# ACT (AI Chat Terminal)
![Main menu](https://github.com/user-attachments/assets/cf6ca3b2-b153-4791-ad0d-6b5abe0ecde6)

A versatile command-line chat interface that supports multiple AI providers including OpenAI, Anthropic, and OpenRouter, featuring rich text formatting, comprehensive file handling capabilities, and an intuitive terminal-based UI.

## Features

### AI Provider Support
- Multiple AI provider integration:
  - OpenAI (GPT-4, GPT-3.5, etc.)
  - Anthropic (Claude models)
  - OpenRouter (access to multiple providers' models)
- Interactive model selection menu with provider-specific features
- Favorite models management for quick access
- Customizable system instructions for AI behavior
- Support for image-enabled models (where available)
- Model context length and pricing information display
- Featured/top model highlighting

### Document Handling
- Support for multiple document formats:
  - Microsoft Word (DOCX)
  - PDF documents
  - OpenDocument Text (ODT)
  - Rich Text Format (RTF)
  - Plain text files with multiple encoding support
  - Source code files with syntax highlighting
- Automatic file type detection and appropriate handling
- Support for files with spaces in names using quotes

### Chat Features
- Interactive chat interface with rich text formatting
- Active system instruction display in chat responses
- Comprehensive help system (`/help` command)
- File and directory reference system:
  - View file contents: `[[ file:path/to/file]]`
  - List directory contents: `[[ dir:path/to/directory]]`
  - View codebase contents: `[[ codebase:path/to/codebase]]`
  - Support for quoted paths: `[[ file:"path with spaces.txt"]]`
- Image embedding support:
  - Local images: `[[ img:path/to/image.jpg]]`
  - URL images: `[[ img:https://example.com/image.jpg]]`
  - Automatic format detection and conversion
  - Support for multiple image formats (JPG, PNG, etc.)
- Chat session management:
  - Save chat history with optional custom name (`/save [name]`)
  - Clear screen and history (`/clear`)
  - Insert multiline text (`/insert`)
  - End chat session (`/end`)
  - Exit options (`exit`, `quit`, `bye`, or Ctrl+C)
  - Automatic chat history saving in both JSON and text formats

### Interface
- Rich console interface with syntax highlighting
- Progress indicators for AI responses
- Comprehensive logging system with rotation
- Detailed error handling and reporting
- Interactive menu system with:
  - Provider categorization
  - Model availability indicators
  - Featured model highlighting
  - Favorite models quick access
- Settings management:
  - Codebase search configuration
  - File type filtering
  - Search depth control
  - Exclusion patterns
  - Appearance customization:
    - AI name color configuration
    - Instruction name color settings
    - Border and text styling options

### Agent Functionality
- Intelligent context-aware AI responses:
  - Semantic search over codebase
  - Automatic file indexing and embedding
  - Context-relevant code suggestions
  - Smart file content retrieval
- Multiple embedding models support:
  - text-embedding-3-small (1536 dimensions)
  - text-embedding-3-large (3072 dimensions)
  - text-embedding-ada-002 (1536 dimensions, legacy)
- ChromaDB integration:
  - Multiple vector stores support
  - Persistent embeddings storage
  - Fast semantic search capabilities
  - Automatic file updates tracking
- Configurable indexing options:
  - Customizable file type filtering
  - Directory exclusion patterns
  - Maximum file size limits
  - Auto-add file capabilities
- Provider compatibility:
  - Full support for OpenAI models
  - Embedding-only mode for OpenRouter
  - Disabled for Anthropic (unsupported)
- Store management features:
  - Create multiple stores
  - Switch between stores
  - Process directories on-demand
  - Test search functionality
  - Delete unused stores
- Agent status indicators:
  - 🟢 Active (Store selected)
  - 🟡 Enabled (No store)
  - ⭕ Disabled
- Advanced settings:
  - Embedding model selection
  - Auto-indexing configuration
  - File size thresholds
  - File type management
  - Exclusion pattern control

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)
- API keys for supported providers:
  - OpenAI API key (required)
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
```env
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
1. Provider selection:
   - OpenAI (indicated by 🟢 when API key is available)
   - Anthropic (indicated by 🟢 when API key is available)
   - OpenRouter (indicated by 🟢 when API key is available)
2. Favorite models management (★ section)
3. Model selection with detailed information:
   - Model name and description
   - Context length information
   - Pricing details (where available)
   - Featured model indicators
4. System instruction customization:
   - Create new instructions
   - Select from existing instructions
   - View and manage instruction sets
5. Chat session initiation
6. Application settings management:
   - Configure codebase search behavior
   - Manage file type inclusions
   - Set search depth preferences
   - Define exclusion patterns
   - Customize appearance settings

### Agent Configuration
The Agent provides context-aware capabilities through vector search:
1. Enable the Agent:
   - Access AI Settings from the main menu
   - Toggle Agent status (Enabled/Disabled)
   - Select or create a vector store
2. Configure the store:
   - Create a new store with custom name
   - Process target directories for indexing
   - Select embedding model for vectorization
   - Configure auto-indexing behavior
3. Manage indexed content:
   - Set file type inclusions (.py, .js, etc.)
   - Define exclusion patterns (node_modules, etc.)
   - Set maximum file size limits
   - Enable/disable auto-add for new files
4. Test and verify:
   - Use the test search functionality
   - Verify relevant context retrieval
   - Adjust settings as needed

When the Agent is active:
- OpenAI models: Full functionality with context-aware responses
- OpenRouter models: Embedding-only mode for search
- Anthropic models: Agent features disabled (unsupported)

Status indicators in the menu show:
- 🟢 Active: Store selected and ready
- 🟡 Enabled: No store selected
- ⭕ Disabled: Agent features off

### Chat Commands
During chat sessions:
- `/help` - Display comprehensive help guide
- `/save [name]` - Save the current chat history with optional custom name
- `/clear` - Clear screen and reset chat history
- `/insert` - Insert multiline text (end with END on new line)
- `/end` - End the chat session
- `exit`, `quit`, `bye`, or Ctrl+C - End the session

### File References
Reference files in your messages:
```
[[ file:example.py]]              # View file contents
[[ file:"path with spaces.txt"]]  # Paths with spaces need quotes
[[ dir:project/src]]              # List directory contents
[[ codebase:src/*.py]]           # View Python files in src
[[ img:image.jpg]]               # Embed local image
[[ img:"https://..."]]           # Embed image from URL
```

### Chat History
Chat histories are automatically saved in both JSON and text formats, organized by:
- Provider (OpenAI, Anthropic, OpenRouter)
- Model company (for OpenRouter)
- Model name
- Custom name (if provided) with timestamp
- Includes:
  - Full conversation history
  - System instructions used
  - Model configuration
  - Timestamps for each message
  - Custom chat name (when specified)
  - Code block formatting in text output
  - Image reference tracking

### Usage Video
[Video](https://www.dropbox.com/scl/fi/9lx7v34zfnghhh8fzt2k4/Screen-Recording-2024-12-06-095730.mp4?rlkey=gyd1glz7rkwv6cnrr3j2u7maf&st=mer3ajni&dl=0)

## License

ACT Community License - See LICENSE file for details 
