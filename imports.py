import os
import sys
import json
import logging
import base64
from logging.handlers import RotatingFileHandler
from datetime import datetime
import time
import dotenv
import rich
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
import inquirer
from inquirer import themes
import requests
from collections import defaultdict
from rich.syntax import Syntax
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
import pprint

# Document handling imports
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import odf
    from odf import text, teletype
    ODF_AVAILABLE = True
except ImportError:
    ODF_AVAILABLE = False

try:
    import rtf
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False

# ChromaDB and LangChain imports
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Conditionally import based on provider
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import openai
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False 