# ACT (AI Chat Terminal) 🤖
![Main menu](https://github.com/user-attachments/assets/cf6ca3b2-b153-4791-ad0d-6b5abe0ecde6)

> A powerful command-line chat interface supporting OpenAI, Anthropic, and OpenRouter, with rich text formatting, comprehensive file handling, and an intuitive terminal UI.

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-ACT%20Community-green.svg)](LICENSE)

</div>

## 📑 Table of Contents
- [Key Features](#-key-features)
  - [AI Integration](#-ai-integration)
  - [RAG System](#-rag-system)
  - [Document Support](#-document-support)
  - [Chat Interface](#-chat-interface)
  - [Chat History](#-chat-history)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [RAG Setup](#rag-setup)
- [Usage](#-usage)
- [License](#-license)
- [Forking the Repository](#-forking-the-repository)

## ✨ Key Features

### 🤖 AI Integration
- **Multiple Providers**
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - OpenRouter (Multi-provider access)
- **Smart Features**
  - Interactive model selection
  - Favorites management
  - Custom system instructions
  - Image-enabled models support
  - Context length optimization
  - Real-time pricing display

### 🧠 RAG System
- **Context-Aware Intelligence**
  - Semantic codebase search
  - Auto file indexing & embedding
  - Smart context retrieval
  - Code-aware suggestions

- **Vector Store Management**
  - Multiple stores support
  - ChromaDB integration
  - Fast semantic search
  - Auto-update tracking

- **Embedding Models**
  | Model | Dimensions | Purpose |
  |-------|------------|---------|
  | text-embedding-3-small | 1536 | Cost-effective option |
  | text-embedding-3-large | 3072 | High-performance tasks |
  | text-embedding-ada-002 | 1536 | Legacy support |

- **Provider Support**
  | Provider | RAG Support | Mode |
  |----------|--------------|------|
  | OpenRouter | ✅ | Full functionality |
  | OpenAI | ✅ | Full functionality |
  | Anthropic | ✅ | Full functionality |

### 📄 Document Support
- **Format Compatibility**
  - Microsoft Word (DOCX)
  - PDF Documents
  - OpenDocument (ODT)
  - Rich Text (RTF)
  - Plain Text (Multiple encodings)
  - Source Code (Syntax highlighting)

### 💬 Chat Interface
- **Rich Formatting**
  - Syntax highlighting
  - Progress indicators
  - Color-coded responses
  - Error handling

- **File References**
  ```
  [[ file:example.py ]]              # View file
  [[ file:"path with spaces.txt" ]]  # Spaces in path
  [[ dir:project/src ]]              # List directory
  [[ codebase:src/*.py ]]           # View codebase
  [[ img:image.jpg ]]               # Local image
  [[ img:"https://..." ]]           # URL image
  ```

- **Commands**
  | Command | Description |
  |---------|-------------|
  | `/help` | Show help guide |
  | `/info` | Display chat session info |
  | `/fav` | Toggle favorite model |
  | `/save [name]` | Save chat history |
  | `/clear` | Clear screen |
  | `/insert` | Multi-line input |
  | `/end` | End session |

### 💾 Chat History
- **Saved Information**
  - Model details (name, ID, provider)
  - System instruction used
  - RAG store and embedding model (if enabled)
  - Total tokens and cost
  - Full conversation history
  - Session duration
  - Date and time

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
- Model management (⭐ for favorites)
- System instruction configuration
- Settings customization
- RAG configuration

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
