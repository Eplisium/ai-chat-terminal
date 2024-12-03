# ACT (AI Chat Terminal)

A versatile command-line chat interface that supports multiple AI providers including OpenAI and OpenRouter.

## Features

- Support for multiple AI providers (OpenAI and OpenRouter)
- Interactive model selection menu
- Favorite models management
- Code and file reference support in chat
- Rich console interface with syntax highlighting
- Comprehensive logging system
- Environment-based configuration

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)
- API keys for OpenAI and/or OpenRouter

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

The interactive menu will guide you through:
1. Selecting an AI provider
2. Choosing a model
3. Starting a chat session

During chat, you can:
- Reference files using `[[file:path/to/file]]`
- View directories using `[[dir:path/to/directory]]`
- View entire codebases using `[[codebase:path/to/codebase]]`
- Save chat history using `/save`
- Clear screen and history using `/clear`
- Exit using 'exit', 'quit', or Ctrl+C

## License

MIT License - See LICENSE file for details 