from imports import *

def setup_logging():
    """
    Set up logging configuration with Rich logging handler
    
    Returns:
        tuple: Logger and Rich Console object
    """
    console = Console()
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Define log format with more details
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    detailed_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

    # Create formatters
    file_formatter = logging.Formatter(detailed_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Create rotating file handler for general logs
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, f'act_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # File shows DEBUG and above

    # Create error file handler for error-only logs
    error_handler = RotatingFileHandler(
        os.path.join(logs_dir, f'act_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)  # Only ERROR and CRITICAL

    # Create response logger for API responses
    response_handler = RotatingFileHandler(
        os.path.join(logs_dir, f'act_responses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    response_handler.setFormatter(file_formatter)
    response_handler.setLevel(logging.INFO)

    # Create response logger
    response_logger = logging.getLogger("ACT.responses")
    response_logger.setLevel(logging.INFO)
    response_logger.addHandler(response_handler)
    
    # Ensure the response logger doesn't propagate to root logger
    response_logger.propagate = False

    # Create null handler for console to suppress output
    null_handler = logging.NullHandler()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    root_logger.addHandler(null_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Create ACT logger as a child of root logger
    logger = logging.getLogger("ACT")
    
    # Test response logger
    response_logger.info("Response logging initialized")
    
    # Log system information at startup (only to files)
    logger.info("="*80)
    logger.info("ACT - AI Chat Terminal Starting")
    logger.info("-"*80)
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Operating System: {os.name} - {sys.platform}")
    logger.info(f"Log Directory: {os.path.abspath(logs_dir)}")
    logger.info("="*80)

    return logger, console

def log_api_response(provider, request_data, response_data, error=None):
    """Log API request and response data"""
    response_logger = logging.getLogger("ACT.responses")
    pp = pprint.PrettyPrinter(indent=2)
    
    log_entry = [
        "="*80,
        f"API CALL: {provider}",
        f"TIMESTAMP: {datetime.now().isoformat()}",
        "-"*80,
        "REQUEST:",
        pp.pformat(request_data),
        "-"*80,
        "RESPONSE:"
    ]
    
    if error:
        log_entry.append("ERROR:")
        log_entry.append(str(error))
    else:
        if provider == "OpenRouter":
            log_entry.append(pp.pformat(response_data))
        elif provider == "OpenAI":
            try:
                response_dict = {
                    "id": response_data.id,
                    "model": response_data.model,
                    "choices": [
                        {
                            "index": c.index,
                            "message": {
                                "role": c.message.role,
                                "content": c.message.content
                            },
                            "finish_reason": c.finish_reason
                        } for c in response_data.choices
                    ],
                    "usage": {
                        "prompt_tokens": response_data.usage.prompt_tokens,
                        "completion_tokens": response_data.usage.completion_tokens,
                        "total_tokens": response_data.usage.total_tokens
                    }
                }
                log_entry.append(pp.pformat(response_dict))
            except Exception as e:
                log_entry.append(f"Error formatting OpenAI response: {str(e)}")
                log_entry.append(str(response_data))
        elif provider == "Anthropic":
            try:
                response_dict = {
                    "id": response_data.id,
                    "type": response_data.type,
                    "role": response_data.role,
                    "content": [
                        {"type": c.type, "text": c.text}
                        for c in response_data.content
                    ],
                    "model": response_data.model,
                    "stop_reason": response_data.stop_reason,
                    "stop_sequence": response_data.stop_sequence,
                    "usage": {
                        "input_tokens": response_data.usage.input_tokens,
                        "output_tokens": response_data.usage.output_tokens
                    }
                }
                log_entry.append(pp.pformat(response_dict))
            except Exception as e:
                log_entry.append(f"Error formatting Anthropic response: {str(e)}")
                log_entry.append(str(response_data))
    
    log_entry.append("="*80 + "\n")
    full_entry = "\n".join(log_entry)
    response_logger.info(full_entry)

def sanitize_filename(filename):
    """Sanitize a filename string to be valid on all operating systems.
    Use this for creating safe filenames, not for path normalization."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    filename = filename.strip('. ')
    if not filename:
        filename = 'unnamed'
    return filename

# Alias for backwards compatibility
sanitize_path = sanitize_filename

def normalize_path(path):
    """Normalize and sanitize file/directory path.
    Handles quotes, backslashes, drive letters, and path normalization."""
    try:
        # Remove any surrounding quotes (both single and double)
        path = path.strip()
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        # Handle Windows paths with escaped backslashes
        if '\\\\' in path:
            path = path.replace('\\\\', '\\')
        
        # Convert forward slashes to backslashes on Windows
        if os.name == 'nt':
            path = path.replace('/', '\\')
        
        # Normalize path separators for the current OS
        path = os.path.normpath(path)
        
        # Make path absolute if relative
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        
        # Ensure Windows drive letter is properly cased
        if os.name == 'nt' and len(path) > 1 and path[1] == ':':
            path = path[0].upper() + path[1:]
        
        return path
    except Exception:
        return path

def read_document_content(file_path):
    """Read content from various document formats"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.docx':
            if not DOCX_AVAILABLE:
                return False, "python-docx library not installed. Install with: pip install python-docx"
            doc = docx.Document(file_path)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return True, content

        elif file_ext == '.pdf':
            if not PDF_AVAILABLE:
                return False, "PyPDF2 library not installed. Install with: pip install PyPDF2"
            reader = PdfReader(file_path)
            content = '\n'.join([page.extract_text() for page in reader.pages])
            return True, content

        elif file_ext == '.odt':
            if not ODF_AVAILABLE:
                return False, "odfpy library not installed. Install with: pip install odfpy"
            doc = odf.load(file_path)
            content = teletype.extractText(doc)
            return True, content

        elif file_ext == '.rtf':
            if not RTF_AVAILABLE:
                return False, "pyth library not installed. Install with: pip install pyth"
            with open(file_path, 'rb') as f:
                doc = rtf.Rtf(f)
                content = doc.getText()
                return True, content

        else:
            encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        return True, content
                except UnicodeDecodeError:
                    continue
            
            return False, f"Could not decode file with any of the following encodings: {', '.join(encodings)}"

    except Exception as e:
        return False, str(e)

def encode_image_to_base64(image_path_or_url):
    """Encode an image to base64 string, handling both local files and URLs"""
    try:
        parsed = urlparse(image_path_or_url)
        is_url = bool(parsed.scheme and parsed.netloc)
        
        if is_url:
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image_data = response.content
        else:
            with open(image_path_or_url, 'rb') as image_file:
                image_data = image_file.read()
        
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"
        
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return True, base64_string
    
    except requests.exceptions.RequestException as e:
        return False, f"Failed to download image: {str(e)}"
    except Exception as e:
        return False, f"Failed to process image: {str(e)}"

def get_image_mime_type(image_path_or_url):
    """Get the MIME type of an image"""
    try:
        parsed = urlparse(image_path_or_url)
        is_url = bool(parsed.scheme and parsed.netloc)
        
        if is_url:
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image_data = response.content
        else:
            with open(image_path_or_url, 'rb') as image_file:
                image_data = image_file.read()
        
        img = Image.open(BytesIO(image_data))
        mime_type = f"image/{img.format.lower()}"
        return mime_type
    
    except Exception:
        return "image/jpeg" 