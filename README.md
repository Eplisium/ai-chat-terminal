# ACT (AI Chat Terminal) 🤖

![Main menu](https://github.com/user-attachments/assets/cf6ca3b2-b153-4791-ad0d-6b5abe0ecde6)

> A powerful command-line chat interface supporting OpenAI, Anthropic, and OpenRouter, with rich text formatting, comprehensive file handling, and an intuitive terminal UI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-ACT%20Community-green.svg)](LICENSE)



## 📑 Table of Contents

- [Key Features](#-key-features)
  - [AI Integration](#-ai-integration)
  - [RAG System](#-rag-system)
  - [AI Tools](#-ai-tools)
  - [Document Support](#-document-support)
  - [Chat Interface](#-chat-interface)
  - [Chat History](#-chat-history)
  - [Statistics & Analytics](#-statistics--analytics)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [RAG Setup](#rag-setup)
- [Usage](#-usage)
  - [Custom API Providers](#custom-api-providers)
- [License](#-license)
- [Forking the Repository](#-forking-the-repository)

## ✨ Key Features

### 🤖 AI Integration

- **Multiple Providers**
  - OpenAI (GPT-4o, GPT-4, GPT-3.5)
  - Anthropic (Claude 3.5, Claude 3)
  - OpenRouter (Access to 80+ AI models)
- **Smart Features**
  - Interactive model selection with company grouping
  - Favorites and recent models management
  - Custom system instructions with templates
  - Image-enabled models support (vision capabilities)
  - Context length optimization
  - Real-time pricing and token tracking

- **Advanced Models Support**

  | Provider | Models | Features |
  |----------|--------|----------|
  | OpenAI | GPT-4o, GPT-4, GPT-3.5 | Vision, RAG, Tools |
  | Anthropic | Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku | Vision, RAG, Tools |
  | OpenRouter | All major provider models | Provider-dependent |

- **Real-time Pricing**

  | Model Type | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
  |------------|------------------------------|------------------------------|
  | GPT-4o | $2.50 | $10.00 |
  | GPT-4o mini | $0.15 | $0.60 |
  | GPT-4 Turbo | $10.00 | $30.00 |
  | GPT-3.5 Turbo | $0.50 | $1.50 |
  | Claude 3.5 Sonnet | $3.00 | $15.00 |
  | Claude 3.5 Haiku | $0.80 | $4.00 |
  | Claude 3 Opus | $15.00 | $75.00 |
  | Claude 3 Sonnet | $3.00 | $15.00 |
  | Claude 3 Haiku | $0.25 | $1.25 |

### 🧠 RAG System

- **Context-Aware Intelligence**
  - Semantic codebase search
  - Auto file indexing & embedding
  - Smart context retrieval
  - Code-aware suggestions
  - Cross-file reference resolution

- **Vector Store Management**
  - Multiple stores creation and selection
  - ChromaDB integration
  - Fast semantic search with relevance scoring
  - Automatic content tracking and updates
  - Document metadata and tagging

- **Embedding Models**

  | Model | Dimensions | Provider | Purpose |
  |-------|------------|----------|---------|
  | text-embedding-3-small | 1536 | OpenAI | Cost-effective option |
  | text-embedding-3-large | 3072 | OpenAI | High-performance tasks |
  | text-embedding-ada-002 | 1536 | OpenAI | Legacy support |
  | snowflake-arctic-embed-l-v2.0 | 1024 | Local (Snowflake) | High performance local model |
  | nomic-embed-text-v2-moe | 768 | Local (Nomic AI) | Mixture of experts local model |

- **Local Embeddings Support**
  - No API key required for local models
  - Automatic download and caching of models
  - GPU acceleration when available (CUDA)
  - Optimized for offline usage scenarios
  - Lower latency for on-premise deployments

- **Provider Support**

  | Provider | RAG Support | Mode | Tools Support |
  |----------|--------------|------|--------------|
  | OpenRouter | ✅ | Full functionality | Model-dependent |
  | OpenAI | ✅ | Full functionality | ✅ |
  | Anthropic | ✅ | Full functionality | ✅ |
  | Custom API | ✅ | Full functionality | ✅ |

- **Advanced Configuration**
  - Custom file type filtering
  - Exclusion patterns
  - Max file size limits
  - Search results customization
  - Auto-add file functionality

### 🛠 AI Tools

- **Built-in Tools**
  - **Calculator**: Perform mathematical calculations directly
  - **Filesystem**: Navigate, read and manage files
  - **Search**: Find content across your system
  - **System**: Access system information and status
  - **Time**: Time-related functions and conversions
  - **Weather**: Retrieve weather information

- **Tools Configuration**
  - Individual tool enabling/disabling
  - Custom tool parameters
  - Tool settings persistence
  - Support across all compatible models

- **Tool Management**
  - Toggle entire tools system on/off
  - Configure individual tools
  - Tool execution tracking
  - Result integration into chat context

### 📄 Document Support

- **Format Compatibility**
  - Microsoft Word (DOCX)
  - PDF Documents
  - OpenDocument (ODT)
  - Rich Text (RTF)
  - Plain Text (Multiple encodings)
  - Source Code (Syntax highlighting)
  - Images (JPG, PNG, WebP, SVG)

- **Advanced File Handling**
  - Recursive directory exploration
  - Smart file type detection
  - Size-aware content loading
  - Cross-file reference resolution
  - Automatic content updating

### 💬 Chat Interface

- **Rich Formatting**
  - Syntax highlighting for code blocks
  - Progress indicators with timing
  - Color-coded responses by provider and model
  - Custom color themes
  - Error handling with detailed feedback

- **File References**

  ```
  [[ file:example.py ]]              # View file
  [[ file:"path with spaces.txt" ]]  # Spaces in path
  [[ dir:project/src ]]              # List directory
  [[ codebase:src/*.py ]]           # View codebase matching pattern
  [[ img:image.jpg ]]               # Include local image
  [[ img:"https://..." ]]           # Include image from URL
  ```

- **Advanced Commands**

  | Command | Description |
  |---------|-------------|
  | `/help` | Show comprehensive help guide |
  | `/info` | Display chat session info, tokens, and cost |
  | `/fav` | Toggle favorite model (add/remove current model) |
  | `/save [name]` | Save chat history with optional custom name |
  | `/clear` | Clear screen and current chat history |
  | `/insert` | Enter multi-line input mode |
  | `/end` | End current chat session |
  | `exit`, `quit`, `bye` | Alternative ways to end session |

- **Multi-line Input Support**
  - Enter complex, formatted text
  - Paste code blocks with proper formatting
  - Special commands: `END` to finish, `CANCEL` to abort
  - Support for programming examples and documentation

- **Keyboard Shortcuts**
  - Ctrl+C: Interrupt AI response
  - Enter: Send message
  - Ctrl+C during prompt: End session

- **Streaming Mode**
  - Real-time token-by-token response streaming
  - Interrupt capability for long responses
  - Response timing display

### 💾 Chat History

- **Saved Information**
  - Model details (name, ID, provider)
  - System instruction used
  - RAG store and embedding model (if enabled)
  - Total tokens and cost breakdown
  - Full conversation history with rich formatting
  - Session duration and timestamps
  - Tool usage records
  - Date and time

- **Export Formats**
  - JSON (structured data with metadata)
  - Plain text (formatted for readability)
  - Automatic file naming with timestamps
  - Custom name support
  - Provider-based organization

- **History Management**
  - Automatic saving with session info
  - Organized by provider and model
  - Search and filtering capabilities
  - Cost and token usage tracking

### 📊 Statistics & Analytics

- **Usage Tracking**
  - Session counts and duration
  - Token usage by model
  - Cost tracking and aggregation
  - Favorite models statistics
  - Command usage frequency

- **Model Analytics**
  - Response time tracking
  - Token efficiency metrics
  - Cost-per-response analysis
  - Most used models ranking

- **Data Management**
  - Clear statistics database option
  - Chat history cleanup
  - Log management

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- API Keys:
  - OpenAI (Required)
  - Anthropic (Optional)
  - OpenRouter (Optional)

### Installation

1. **Clone Repository**

   ```bash
   git clone https://github.com/eplisium/ai-chat-terminal.git
   cd ai-chat-terminal
   ```

2. **Setup Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   Create `.env` file:

   ```env
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```

### RAG Setup

1. **Enable RAG**
   - Open AI Settings menu
   - Toggle RAG status
   - Create/select vector store

2. **Configure Store**
   - Name your store
   - Process target directories
   - Select embedding model
   - Set indexing preferences

3. **Manage Content**
   - Configure file types
   - Set exclusion patterns
   - Adjust size limits
   - Enable auto-indexing

4. **Status Indicators**

   | Icon | Status | Description |
   |------|--------|-------------|
   | 🟢 | Active | Store ready |
   | 🟡 | Enabled | No store selected |
   | ⭕ | Disabled | Features off |

## 📖 Usage

Run the application:

```bash
python main.py
```

The interface provides:

- Provider selection (🟢 indicates available)
- Model management (⭐ for favorites, 🕒 for recent)
- System instruction configuration
- Settings customization
- RAG configuration
- AI tools management
- Statistics and analytics

### Custom API Providers

ACT supports integration with custom API providers that follow the OpenAI-compatible API format:

1. **Adding a Custom Provider**
   - Select "Manage Custom APIs" from the main menu
   - Click "Add New Provider"
   - Enter provider details:
     - Name: A display name for the provider
     - API Base URL: The base URL for the API (e.g., `http://localhost:10001/v1`)
     - API Key: Optional authentication key
     - Use Authorization Header: Whether to send API key in Authorization header
     - Append '/chat/completions' to base URL: Toggle whether to use `/chat/completions` endpoint

2. **Endpoint Configuration**
   - If your API already includes the complete path: Disable "Append '/chat/completions'"
   - If your API expects the standard OpenAI path structure: Enable "Append '/chat/completions'"
   
3. **Error Troubleshooting**
   - If you get "404 Not Found" errors for the URL with `/chat/completions`, try:
     - Edit the provider and disable the "Append '/chat/completions'" option
     - Ensure the base URL already includes any necessary path components

4. **Model Configuration**
   - Add one or more models for each provider
   - Specify model ID as expected by the API
   - Set context window and max token values appropriate for your model

5. **Provider Management**
   - Edit provider details and models at any time
   - Test API connections with the selected model
   - Remove providers when no longer needed

This flexibility allows integration with local models, self-hosted servers, and custom API implementations that follow the OpenAI format.

## 📝 License

ACT Community License - See [LICENSE](LICENSE) file

## 🍴 Forking the Repository

### How to Fork

To fork this repository, follow these steps:

1. Go to the repository page on GitHub: [Eplisium/ai-chat-terminal](https://github.com/eplisium/ai-chat-terminal).
2. Click the "Fork" button at the top right of the page.
3. Choose your GitHub account or organization where you want to create the fork.
4. After the fork is created, you can clone it to your local machine using the URL of your forked repository.

### Benefits of Forking

Forking this repository allows you to:

- Experiment with the code without affecting the original project.
- Contribute to the project by submitting pull requests.
- Customize the project to fit your specific needs.
