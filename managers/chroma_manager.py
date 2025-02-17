from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import errno
from dotenv import load_dotenv
import json
import torch
from rich.console import Console
from rich.panel import Panel
import logging
import inquirer
import glob
import datetime
import time
import uuid

class FileContent(BaseModel):
    """Model for file content"""
    filename: str = Field(..., description="Name of the file")
    content: str = Field(..., description="Content of the file")
    file_type: str = Field(..., description="Type/extension of the file")

class ChromaManager:
    # Available embedding models and their dimensions
    EMBEDDING_MODELS = {
        'text-embedding-3-small': {'dimensions': 1536, 'description': 'OpenAI - Smallest and most cost-effective model', 'type': 'openai'},
        'text-embedding-3-large': {'dimensions': 3072, 'description': 'OpenAI - Most capable model for complex tasks', 'type': 'openai'},
        'text-embedding-ada-002': {'dimensions': 1536, 'description': 'OpenAI - Legacy model, good balance of performance and cost', 'type': 'openai'},
        'snowflake-arctic-embed-l-v2.0': {'dimensions': 1024, 'description': 'Snowflake - High performance local model', 'type': 'local'},
        'nomic-embed-text-v2-moe': {'dimensions': 768, 'description': 'Nomic AI - Mixture of experts local model', 'type': 'local'}
    }

    def __init__(self, logger: logging.Logger, console: Console):
        """Initialize ChromaDB Manager"""
        self.logger = logger
        self.console = console
        self.embeddings = None
        self.vectorstore = None
        self.store_name = None
        self.current_directory = None  # Track current directory
        self.persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chroma_stores')
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
        self.embedding_model_name = None  # Track current embedding model name
        self.model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_cache')
        
        # Create necessary directories
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize embeddings
        self.initialize_embeddings()
    
    def initialize_embeddings(self) -> None:
        """Initialize embeddings with the selected model"""
        try:
            # Load settings to get the selected embedding model
            settings = self._load_settings()
            selected_model = settings.get('chromadb', {}).get('embedding_model', 'text-embedding-3-small')
            self.embedding_model_name = selected_model

            model_info = self.EMBEDDING_MODELS.get(selected_model)
            if not model_info:
                raise ValueError(f"Unknown embedding model: {selected_model}")

            if model_info['type'] == 'openai':
                # Initialize OpenAI embeddings
                self.openai_api_key = os.getenv('OPENAI_API_KEY')
                if not self.openai_api_key:
                    self.logger.warning("OpenAI API key not found in environment variables")
                    self.console.print(Panel(
                        "[yellow]OpenAI API key not found. Please add it to your .env file:[/yellow]\n\n"
                        "OPENAI_API_KEY=your-api-key-here",
                        title="Configuration Required",
                        border_style="yellow"
                    ))
                    return

                self.embeddings = OpenAIEmbeddings(
                    model=selected_model,
                    openai_api_key=self.openai_api_key,
                    chunk_size=1000,
                    max_retries=3
                )
            else:  # local models
                model_name = f"Snowflake/{selected_model}" if "snowflake" in selected_model else f"nomic-ai/{selected_model}"
                model_kwargs = {
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
                
                # Add trust_remote_code for Nomic model
                if "nomic" in selected_model:
                    model_kwargs['trust_remote_code'] = True
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs={'normalize_embeddings': True},
                    cache_folder=self.model_cache_dir
                )
            
            self.logger.info(f"Successfully initialized embeddings with model: {selected_model}")
            self.console.print(f"[green]Successfully initialized embeddings with model: {selected_model}[/green]")
                
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = None
            self.embedding_model_name = None
            self.console.print(Panel(
                f"[red]Error initializing embeddings:[/red]\n{str(e)}\n\n"
                "[yellow]Please check:[/yellow]\n"
                "1. Your internet connection (for downloading models)\n"
                "2. You have sufficient disk space for local models\n"
                "3. Your system meets the model requirements",
                title="Initialization Error",
                border_style="red"
            ))

    def test_embeddings(self) -> bool:
        """Test if embeddings are working correctly"""
        try:
            if not self.embeddings:
                self.initialize_embeddings()
                if not self.embeddings:
                    return False

            self.console.print("[cyan]Testing embeddings...[/cyan]")
            _ = self.embeddings.embed_query("Test query")
            self.console.print("[green]Embeddings test successful![/green]")
            return True
        except Exception as e:
            self.logger.error(f"Failed to test embeddings: {e}")
            self.embeddings = None
            self.console.print(f"[red]Embeddings test failed: {str(e)}[/red]")
            return False

    def _update_store_metadata(self, directory_path: str = None, files: List[str] = None) -> None:
        """Update store metadata with processed directories and files"""
        if not self.store_name:
            return

        store_path = os.path.join(self.persist_directory, self.store_name)
        metadata_path = os.path.join(store_path, 'metadata.json')
        
        try:
            # Load existing metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Preserve essential store information
            if 'name' not in metadata:
                metadata['name'] = self.store_name
            if 'embedding_model' not in metadata:
                metadata['embedding_model'] = self.embedding_model_name
            if 'embedding_dimensions' not in metadata and self.embedding_model_name:
                metadata['embedding_dimensions'] = self.EMBEDDING_MODELS[self.embedding_model_name]['dimensions']
            if 'created_at' not in metadata:
                metadata['created_at'] = datetime.datetime.now().isoformat()

            # Initialize or update history
            if 'history' not in metadata:
                metadata['history'] = {
                    'directories': [],
                    'files': [],
                    'last_processed': None,
                    'processing_log': []
                }

            # Update with new directory
            if directory_path:
                abs_path = os.path.abspath(directory_path)
                if abs_path not in metadata['history']['directories']:
                    metadata['history']['directories'].append(abs_path)
                metadata['last_directory'] = abs_path  # Keep for backward compatibility
                # Log directory processing
                metadata['history']['processing_log'].append({
                    'type': 'directory',
                    'path': abs_path,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'status': 'added'
                })

            # Update with new files
            if files:
                abs_files = [os.path.abspath(f) for f in files if os.path.exists(f)]
                for file in abs_files:
                    if file not in metadata['history']['files']:
                        metadata['history']['files'].append(file)
                        # Log file processing
                        metadata['history']['processing_log'].append({
                            'type': 'file',
                            'path': file,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'status': 'added'
                        })

            # Update timestamp
            metadata['history']['last_processed'] = datetime.datetime.now().isoformat()

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")

    def refresh_store(self) -> bool:
        """Refresh all contents of the current store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False

            self.console.print("[cyan]Starting store refresh...[/cyan]")

            # Get current store name and metadata
            current_store_name = self.store_name
            store_path = os.path.join(self.persist_directory, current_store_name)
            metadata_path = os.path.join(store_path, 'metadata.json')
            
            # Load existing metadata
            existing_metadata = None
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        existing_metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading existing metadata: {e}")
                return False

            if not existing_metadata or 'history' not in existing_metadata:
                self.console.print("[yellow]No history found in store metadata[/yellow]")
                return False

            # Get all tracked files
            tracked_files = existing_metadata['history'].get('files', [])

            self.console.print(f"[cyan]Found {len(tracked_files)} files to process[/cyan]")

            all_files = []
            processed_paths = set()

            # Process all tracked files
            for file_path in tracked_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            filename = os.path.basename(file_path)
                            file_content = FileContent(
                                filename=filename,
                                content=content,
                                file_type=os.path.splitext(filename)[1]
                            )
                            all_files.append((file_content, os.path.dirname(file_path)))
                            processed_paths.add(file_path)
                            self.console.print(f"[cyan]Processed file: {filename}[/cyan]")
                    except Exception as e:
                        self.logger.error(f"Error reading {file_path}: {str(e)}")
                        self.console.print(f"[yellow]Error reading file: {file_path}[/yellow]")
                else:
                    self.console.print(f"[yellow]File no longer exists: {file_path}[/yellow]")

            if all_files:
                # Clear existing documents
                self.console.print("[cyan]Clearing existing documents...[/cyan]")
                self.clear_store()
                
                # Create documents for ChromaDB
                self.console.print("[cyan]Creating embeddings for files...[/cyan]")
                from langchain_core.documents import Document
                
                documents = []
                for file_content, directory in all_files:
                    doc = Document(
                        page_content=f"File: {file_content.filename}\nContent:\n{file_content.content}",
                        metadata={
                            'filename': file_content.filename,
                            'file_type': file_content.file_type,
                            'directory': directory,
                            'added_at': datetime.datetime.now().isoformat(),
                            'file_size': len(file_content.content.encode('utf-8')),
                            'source': 'refresh',
                            'id': str(uuid.uuid4())
                        }
                    )
                    documents.append(doc)
                
                # Update the store with new documents
                self.vectorstore.add_documents(documents)
                
                # Update metadata
                self._update_store_metadata()
                
                self.console.print(f"[green]Successfully refreshed store with {len(all_files)} files[/green]")
                return True
            else:
                self.console.print("[yellow]No files found to refresh[/yellow]")
                return True

        except Exception as e:
            self.logger.error(f"Error refreshing store: {str(e)}")
            self.console.print(f"[red]Error refreshing store: {str(e)}[/red]")
            return False

    def _load_settings(self) -> Dict:
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                    # Ensure chromadb settings exist with defaults
                    if 'chromadb' not in settings:
                        settings['chromadb'] = {
                            'embedding_model': 'text-embedding-3-small',
                            'auto_add_files': True,
                            'max_file_size_mb': 5,
                            'search_results_limit': 10,  # Add default search limit
                            'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__', 'build', 'dist', 'chroma_stores'],
                            'file_types': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.md', '.txt']
                        }
                        self._save_settings(settings)
                    elif 'search_results_limit' not in settings['chromadb']:
                        settings['chromadb']['search_results_limit'] = 10
                        self._save_settings(settings)
                    return settings
            return {
                'chromadb': {
                    'embedding_model': 'text-embedding-3-small',
                    'auto_add_files': True,
                    'max_file_size_mb': 5,
                    'search_results_limit': 10,
                    'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__', 'build', 'dist', 'chroma_stores'],
                    'file_types': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.md', '.txt']
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return {'chromadb': {'embedding_model': 'text-embedding-3-small', 'search_results_limit': 10}}

    def _save_settings(self, settings: Dict) -> None:
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def read_directory(self, directory_path: str, include_subdirs: bool = True) -> List[FileContent]:
        """Read all files in the directory matching the configured patterns"""
        try:
            settings = self._load_settings()
            chromadb_settings = settings.get('chromadb', {})
            file_types = chromadb_settings.get('file_types', [])
            exclude_patterns = chromadb_settings.get('exclude_patterns', [])
            max_size_mb = chromadb_settings.get('max_file_size_mb', 5)
            
            files = []
            processed_paths = set()  # Track processed paths to avoid duplicates
            
            # Clean and normalize directory path
            directory_path = directory_path.strip('"\'')  # Remove quotes if present
            directory_path = os.path.abspath(directory_path)
            if not os.path.exists(directory_path):
                self.console.print(f"[yellow]Directory not found: {directory_path}[/yellow]")
                return []
            
            self.console.print(f"[cyan]Reading files from: {directory_path}[/cyan]")
            
            # Create glob patterns for each file type
            for file_type in file_types:
                pattern = os.path.join(directory_path, f"**/*{file_type}" if include_subdirs else f"*{file_type}")
                try:
                    for filepath in glob.glob(pattern, recursive=include_subdirs):
                        try:
                            # Skip if already processed
                            if filepath in processed_paths:
                                continue
                            
                            # Convert to absolute path and normalize
                            abs_path = os.path.abspath(filepath)
                            
                            # Skip if path contains any exclude pattern
                            if any(exclude in abs_path for exclude in exclude_patterns):
                                self.logger.debug(f"Skipping excluded path: {abs_path}")
                                continue
                            
                            # Check file size
                            try:
                                file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
                                if file_size_mb > max_size_mb:
                                    self.logger.warning(f"Skipping {abs_path}: File too large ({file_size_mb:.1f}MB > {max_size_mb}MB)")
                                    continue
                            except OSError as e:
                                self.logger.error(f"Error checking file size for {abs_path}: {e}")
                                continue
                            
                            # Try different encodings
                            content = None
                            encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
                            for encoding in encodings:
                                try:
                                    with open(abs_path, 'r', encoding=encoding) as f:
                                        content = f.read()
                                        break
                                except UnicodeDecodeError:
                                    continue
                                except Exception as e:
                                    self.logger.error(f"Error reading {abs_path} with {encoding}: {e}")
                                    break
                            
                            if content is not None:
                                filename = os.path.basename(abs_path)
                                file_type = os.path.splitext(filename)[1]
                                files.append(FileContent(
                                    filename=filename,
                                    content=content,
                                    file_type=file_type
                                ))
                                processed_paths.add(abs_path)
                                self.console.print(f"[cyan]Read: {filename} ({file_size_mb:.1f}MB)[/cyan]")
                            else:
                                self.logger.warning(f"Could not read {abs_path} with any supported encoding")
                                
                        except Exception as e:
                            self.logger.error(f"Error processing {filepath}: {str(e)}")
                            continue
                            
                except Exception as e:
                    self.logger.error(f"Error with glob pattern {pattern}: {str(e)}")
                    continue
            
            if not files:
                self.console.print("[yellow]No valid files found in directory[/yellow]")
            else:
                self.console.print(f"[green]Successfully read {len(files)} files[/green]")
            
            return files
            
        except Exception as e:
            self.logger.error(f"Error reading directory: {str(e)}")
            self.console.print(f"[red]Error reading directory: {str(e)}[/red]")
            return []

    def process_directory(self, directory_path: str, include_subdirs: bool = True, force_refresh: bool = False, skip_subdirs_prompt: bool = False) -> bool:
        """Process all files in a directory and add them to the current store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            # Clean and normalize directory path
            directory_path = directory_path.strip('"\'')  # Remove quotes if present
            directory_path = os.path.abspath(directory_path)
            if not os.path.exists(directory_path):
                self.console.print(f"[yellow]Directory not found: {directory_path}[/yellow]")
                return False

            # If include_subdirs is True and not skipping prompt, let user select which subdirectories to include
            selected_dirs = [directory_path]  # Always include the root directory
            if include_subdirs and not skip_subdirs_prompt:
                # Get list of subdirectories
                subdirs = []
                for root, dirs, _ in os.walk(directory_path):
                    for d in dirs:
                        full_path = os.path.join(root, d)
                        rel_path = os.path.relpath(full_path, directory_path)
                        subdirs.append((rel_path, full_path))
                
                if subdirs:
                    # Let user select which subdirectories to include
                    questions = [
                        inquirer.Checkbox('selected_subdirs',
                            message="Select subdirectories to include",
                            choices=[('(root directory)', directory_path)] + [(d[0], d[1]) for d in subdirs],
                            default=[directory_path]
                        ),
                    ]
                    subdir_answer = inquirer.prompt(questions)
                    
                    if subdir_answer and subdir_answer['selected_subdirs']:
                        selected_dirs = subdir_answer['selected_subdirs']
            elif include_subdirs:
                # If skipping prompt but include_subdirs is True, include all subdirectories
                for root, dirs, _ in os.walk(directory_path):
                    for d in dirs:
                        selected_dirs.append(os.path.join(root, d))
            
            # Process each selected directory
            all_files = []
            processed_paths = set()  # Track processed paths to avoid duplicates
            
            # Save the root directory path and update metadata
            if self.store_name:
                self._update_store_metadata(directory_path=directory_path)
            
            # Read files from root directory first
            files = self.read_directory(directory_path, include_subdirs=False)
            for file in files:
                file_path = os.path.join(directory_path, file.filename)
                if file_path not in processed_paths:
                    doc = Document(
                        page_content=f"File: {file.filename}\nContent:\n{file.content}",
                        metadata={
                            'filename': file.filename,
                            'file_type': file.file_type,
                            'directory': directory_path,
                            'added_at': datetime.datetime.now().isoformat(),
                            'file_size': len(file.content.encode('utf-8')),
                            'source': 'directory_process',
                            'id': str(uuid.uuid4())
                        }
                    )
                    all_files.append(doc)
                    processed_paths.add(file_path)
                    # Update metadata for individual files
                    self._update_store_metadata(files=[file_path])
            
            # Process selected subdirectories
            for dir_path in selected_dirs:
                if dir_path != directory_path:  # Skip root directory as it's already processed
                    # Save the directory path and update metadata
                    if self.store_name:
                        self._update_store_metadata(directory_path=dir_path)
                    
                    # Read files from this directory
                    files = self.read_directory(dir_path, include_subdirs=False)
                    for file in files:
                        file_path = os.path.join(dir_path, file.filename)
                        if file_path not in processed_paths:
                            doc = Document(
                                page_content=f"File: {file.filename}\nContent:\n{file.content}",
                                metadata={
                                    'filename': file.filename,
                                    'file_type': file.file_type,
                                    'directory': dir_path,
                                    'added_at': datetime.datetime.now().isoformat(),
                                    'file_size': len(file.content.encode('utf-8')),
                                    'source': 'directory_process',
                                    'id': str(uuid.uuid4())
                                }
                            )
                            all_files.append(doc)
                            processed_paths.add(file_path)
                            # Update metadata for individual files
                            self._update_store_metadata(files=[file_path])
            
            if all_files:
                # Store in ChromaDB with metadata
                try:
                    self.vectorstore.add_documents(all_files)
                    self.console.print(f"[green]Successfully added {len(all_files)} files to store[/green]")
                    return True
                except Exception as e:
                    self.logger.error(f"Error adding documents to store: {e}")
                    self.console.print(f"[red]Error adding documents to store: {str(e)}[/red]")
                    return False
            else:
                self.console.print("[yellow]No valid files found to process[/yellow]")
                return False
            
        except Exception as e:
            self.logger.error(f"Error processing directory: {str(e)}")
            self.console.print(f"[red]Error processing directory: {str(e)}[/red]")
            return False

    @staticmethod
    def sanitize_store_name(name: str) -> str:
        """
        Sanitize store name to comply with ChromaDB collection name requirements:
        1. 3-63 characters
        2. Starts and ends with alphanumeric
        3. Contains only alphanumeric, underscores or hyphens
        4. No consecutive periods
        """
        # Replace spaces and invalid characters with underscores
        sanitized = ''.join(c if c.isalnum() else '_' for c in name)
        
        # Ensure it starts and ends with alphanumeric
        if not sanitized[0].isalnum():
            sanitized = 'a' + sanitized
        if not sanitized[-1].isalnum():
            sanitized = sanitized + '0'
            
        # Ensure minimum length of 3
        while len(sanitized) < 3:
            sanitized += '0'
            
        # Truncate to maximum length of 63
        sanitized = sanitized[:63]
        
        return sanitized
    
    def list_stores(self) -> List[str]:
        """List all available ChromaDB stores"""
        if not os.path.exists(self.persist_directory):
            return []
        return [d for d in os.listdir(self.persist_directory) 
                if os.path.isdir(os.path.join(self.persist_directory, d))]
    
    def load_store(self, store_name: str) -> bool:
        """Load an existing ChromaDB store"""
        try:
            # Handle "None" or empty store name as special case to unload
            if not store_name or store_name.lower() == "none":
                if self.vectorstore:
                    self.unload_store()
                # Update settings to reflect no store selected
                settings = self._load_settings()
                if 'agent' not in settings:
                    settings['agent'] = {}
                settings['agent']['last_store'] = None
                self._save_settings(settings)
                return True
            
            store_path = os.path.join(self.persist_directory, store_name)
            if not os.path.exists(store_path):
                self.logger.error(f"Store '{store_name}' does not exist")
                return False
            
            # Read store metadata to get the embedding model
            metadata_path = os.path.join(store_path, 'metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                store_model = metadata.get('embedding_model', 'text-embedding-3-small')
            except Exception as e:
                self.logger.error(f"Error reading store metadata: {e}")
                return False
            
            # Check if we need to switch embedding models
            settings = self._load_settings()
            current_model = settings.get('chromadb', {}).get('embedding_model', 'text-embedding-3-small')
            
            if current_model != store_model:
                self.console.print(f"[yellow]Store '{store_name}' uses {store_model} model. Switching embedding model...[/yellow]")
                
                # Update settings with store's model
                if 'chromadb' not in settings:
                    settings['chromadb'] = {}
                settings['chromadb']['embedding_model'] = store_model
                self._save_settings(settings)
                
                # Reinitialize embeddings with store's model
                self.initialize_embeddings()
                if not self.embeddings:
                    self.logger.error("Failed to initialize embeddings with store's model")
                    return False
                
                self.embedding_model_name = store_model  # Update model name after successful initialization
            
            # Unload current store if one is loaded
            if self.vectorstore:
                self.unload_store()
            
            # Create new Chroma instance
            try:
                self.vectorstore = Chroma(
                    persist_directory=store_path,
                    embedding_function=self.embeddings,
                    collection_name=store_name
                )
                self.store_name = store_name
                
                # Test the store by getting document count
                count = self.vectorstore._collection.count()
                self.logger.info(f"Successfully loaded store '{store_name}' with {count} documents using {store_model} embeddings")
                self.console.print(f"[green]Successfully loaded store: {store_name} ({count} documents) using {store_model} embeddings[/green]")
                
                # Update settings with the newly loaded store
                settings = self._load_settings()
                if 'agent' not in settings:
                    settings['agent'] = {}
                settings['agent']['last_store'] = store_name
                self._save_settings(settings)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error initializing store: {e}")
                self.vectorstore = None
                self.store_name = None
                raise
                
        except Exception as e:
            self.logger.error(f"Error loading store: {e}")
            self.console.print(f"[red]Error loading store: {str(e)}[/red]")
            return False
    
    def create_store(self, store_name: str) -> bool:
        """Create a new ChromaDB store"""
        try:
            if not self.embeddings:
                self.logger.error("OpenAI embeddings not available")
                self.console.print(Panel(
                    "[red]OpenAI embeddings not available. Please check:[/red]\n\n"
                    "1. OpenAI API key is set in your .env file\n"
                    "2. Selected embedding model is valid\n"
                    "3. You have access to the selected model",
                    title="Configuration Error",
                    border_style="red"
                ))
                return False
            
            sanitized_name = self.sanitize_store_name(store_name)
            store_path = os.path.join(self.persist_directory, sanitized_name)
            
            if os.path.exists(store_path):
                self.logger.warning(f"Store '{store_name}' already exists")
                return self.load_store(sanitized_name)
            
            os.makedirs(store_path, exist_ok=True)
            
            # Get current embedding model
            settings = self._load_settings()
            current_model = settings.get('chromadb', {}).get('embedding_model', 'text-embedding-3-small')
            
            # Save store metadata with embedding model information
            metadata_path = os.path.join(store_path, 'metadata.json')
            metadata = {
                'name': store_name,
                'created_at': datetime.datetime.now().isoformat(),
                'embedding_model': current_model,
                'embedding_dimensions': self.EMBEDDING_MODELS[current_model]['dimensions'],
                'history': {
                    'directories': [],
                    'files': [],
                    'last_processed': None,
                    'processing_log': []
                }
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.store_name = sanitized_name
            self.vectorstore = Chroma(
                persist_directory=store_path,
                embedding_function=self.embeddings,
                collection_name=sanitized_name
            )
            
            self.console.print(f"[green]Successfully created store: {store_name} with {current_model} embeddings[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating store: {e}")
            self.console.print(f"[red]Error creating store: {str(e)}[/red]")
            return False
    
    def add_file_to_store(self, filepath: str) -> bool:
        """Add a file's content to the current store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            # Normalize path and check file
            abs_path = os.path.abspath(filepath)
            if not os.path.exists(abs_path):
                self.console.print(f"[yellow]File not found: {abs_path}[/yellow]")
                return False
            
            # Check file size
            settings = self._load_settings()
            max_size_mb = settings.get('chromadb', {}).get('max_file_size_mb', 5)
            file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                self.console.print(f"[yellow]File too large: {file_size_mb:.1f}MB > {max_size_mb}MB[/yellow]")
                return False
            
            # Try different encodings
            content = None
            encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    with open(abs_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error reading file with {encoding}: {e}")
                    break
            
            if content is None:
                self.console.print("[yellow]Could not read file with any supported encoding[/yellow]")
                return False
            
            # Create document with metadata
            filename = os.path.basename(abs_path)
            file_type = os.path.splitext(filename)[1]
            
            from langchain_core.documents import Document
            doc = Document(
                page_content=f"File: {filename}\nContent:\n{content}",
                metadata={
                    'filename': filename,
                    'file_type': file_type,
                    'directory': os.path.dirname(abs_path),
                    'added_at': datetime.datetime.now().isoformat(),
                    'file_size': len(content.encode('utf-8')),
                    'source': 'single_file_add',
                    'id': str(uuid.uuid4())
                }
            )
            
            # Add to ChromaDB
            try:
                self.vectorstore.add_documents([doc])
                # Update metadata
                self._update_store_metadata(files=[abs_path])
                self.console.print(f"[green]Successfully added file: {filename}[/green]")
                return True
            except Exception as e:
                self.logger.error(f"Error adding file to store: {e}")
                self.console.print(f"[red]Error adding file to store: {str(e)}[/red]")
                return False
            
        except Exception as e:
            self.logger.error(f"Error adding file to store: {e}")
            self.console.print(f"[red]Error adding file to store: {str(e)}[/red]")
            return False
    
    def search_context(self, query: str, k: int = None) -> List[str]:
        """Search for relevant context in the current store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return []
            
            # Use configured limit if k is not specified
            if k is None:
                settings = self._load_settings()
                k = settings.get('chromadb', {}).get('search_results_limit', 10)
            
            results = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            self.logger.error(f"Error searching context: {e}")
            return []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the current store with their IDs"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return []
            
            collection = self.vectorstore._collection
            result = collection.get()
            
            documents = []
            for i, doc in enumerate(result['documents']):
                doc_id = result['ids'][i] if 'ids' in result else f"doc_{i}"
                metadata = result['metadatas'][i] if 'metadatas' in result else {}
                documents.append({
                    'id': doc_id,
                    'content': doc,
                    'metadata': metadata
                })
            
            return documents
        except Exception as e:
            self.logger.error(f"Error getting documents: {e}")
            return []

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete specific documents from the store by their IDs"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            self.vectorstore._collection.delete(ids=doc_ids)
            self.console.print(f"[green]Successfully deleted {len(doc_ids)} documents[/green]")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            self.console.print(f"[red]Error deleting documents: {str(e)}[/red]")
            return False

    def update_document(self, doc_id: str, new_content: str) -> bool:
        """Update a specific document in the store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            # Create new embedding for the updated content
            self.vectorstore._collection.update(
                ids=[doc_id],
                documents=[new_content]
            )
            self.console.print(f"[green]Successfully updated document: {doc_id}[/green]")
            return True
        except Exception as e:
            self.logger.error(f"Error updating document: {e}")
            self.console.print(f"[red]Error updating document: {str(e)}[/red]")
            return False

    def clear_store(self) -> bool:
        """Remove all documents from the current store while preserving metadata"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            # Get all document IDs
            documents = self.get_all_documents()
            if not documents:
                return True  # Store is already empty
            
            doc_ids = [doc['id'] for doc in documents]
            self.vectorstore._collection.delete(ids=doc_ids)
            
            # Log the clearing operation in metadata
            if self.store_name:
                self._update_store_metadata()  # Just update timestamp
                store_path = os.path.join(self.persist_directory, self.store_name)
                metadata_path = os.path.join(store_path, 'metadata.json')
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['history']['processing_log'].append({
                        'type': 'clear',
                        'timestamp': datetime.datetime.now().isoformat(),
                        'status': 'success',
                        'documents_cleared': len(doc_ids)
                    })
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                except Exception as e:
                    self.logger.error(f"Error updating metadata after clear: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error clearing store: {e}")
            self.console.print(f"[red]Error clearing store: {str(e)}[/red]")
            return False
    
    def delete_store(self, store_name: str) -> bool:
        """Delete a ChromaDB store"""
        try:
            store_path = os.path.join(self.persist_directory, store_name)
            if not os.path.exists(store_path):
                self.logger.error(f"Store '{store_name}' does not exist")
                return False
            
            # If this is the current store, unload it first
            if self.store_name == store_name:
                self.unload_store()
            
            # Update settings to remove this store if it was the last used
            settings = self._load_settings()
            if settings.get('agent', {}).get('last_store') == store_name:
                if 'agent' not in settings:
                    settings['agent'] = {}
                settings['agent']['last_store'] = None
                self._save_settings(settings)
            
            # Force cleanup of any potential handles
            import gc
            gc.collect()  # Force garbage collection
            time.sleep(1)  # Give more time for resources to be released
            
            import shutil
            max_retries = 5
            retry_delay = 1  # Start with longer delay
            
            def handle_error(e: Exception, path: str, is_file: bool = True) -> None:
                """Handle deletion errors with detailed logging"""
                import errno
                if isinstance(e, PermissionError):
                    self.logger.warning(f"Permission denied for {'file' if is_file else 'directory'} {path}")
                elif isinstance(e, OSError) and e.errno == errno.EACCES:
                    self.logger.warning(f"Access denied for {'file' if is_file else 'directory'} {path}")
                elif isinstance(e, OSError) and e.errno == errno.EBUSY:
                    self.logger.warning(f"{'File' if is_file else 'Directory'} {path} is in use")
                else:
                    self.logger.warning(f"Error deleting {'file' if is_file else 'directory'} {path}: {str(e)}")
            
            for attempt in range(max_retries):
                try:
                    # Try to remove any read-only attributes on Windows
                    if os.name == 'nt':
                        import stat
                        def remove_readonly(func, path, excinfo):
                            try:
                                os.chmod(path, stat.S_IWRITE)
                                func(path)
                            except Exception as e:
                                handle_error(e, path)
                        
                        # Apply to all files first
                        for root, dirs, files in os.walk(store_path):
                            for fname in files:
                                full_path = os.path.join(root, fname)
                                try:
                                    os.chmod(full_path, stat.S_IWRITE)
                                except Exception as e:
                                    handle_error(e, full_path)
                    
                    # Try to delete the directory with error handler
                    if os.path.exists(store_path):
                        shutil.rmtree(store_path, onerror=remove_readonly if os.name == 'nt' else None)
                    
                    # Verify deletion
                    if not os.path.exists(store_path):
                        self.logger.info(f"Successfully deleted store: {store_name}")
                        self.console.print(f"[green]Successfully deleted store: {store_name}[/green]")
                        return True
                    
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retry {attempt + 1}/{max_retries}: Directory still exists")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise PermissionError(f"Could not delete store directory after {max_retries} attempts")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retry {attempt + 1}/{max_retries}: Error while deleting store: {str(e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting store: {e}")
            self.console.print(f"[red]Error deleting store: {str(e)}[/red]")
            return False

    def select_embedding_model(self) -> bool:
        """Allow user to select an embedding model"""
        try:
            settings = self._load_settings()
            current_model = settings.get('chromadb', {}).get('embedding_model', 'text-embedding-3-small')

            # Create choices for the models
            choices = []
            for model_name, info in self.EMBEDDING_MODELS.items():
                is_current = "✓ " if model_name == current_model else "  "
                choices.append((
                    f"{is_current}{model_name} - {info['description']} ({info['dimensions']} dimensions)",
                    model_name
                ))
            choices.append(("Back", None))

            questions = [
                inquirer.List('model',
                    message="Select Embedding Model",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['model'] is None:
                return False

            selected_model = answer['model']
            
            # Warn if a store is loaded with a different model
            if self.store_name:
                store_path = os.path.join(self.persist_directory, self.store_name)
                metadata_path = os.path.join(store_path, 'metadata.json')
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    store_model = metadata.get('embedding_model')
                    if store_model and store_model != selected_model:
                        self.console.print(
                            f"[yellow]Warning: Current store uses {store_model} embeddings. "
                            "You'll need to refresh the store to use the new model.[/yellow]"
                        )
                except Exception as e:
                    self.logger.warning(f"Could not read store metadata: {e}")
            
            # Update settings
            if 'chromadb' not in settings:
                settings['chromadb'] = {}
            settings['chromadb']['embedding_model'] = selected_model
            self._save_settings(settings)

            # Reinitialize embeddings with new model
            self.initialize_embeddings()
            if self.embeddings:
                self.embedding_model_name = selected_model  # Update model name after successful initialization
                self.console.print(f"[green]Successfully updated embedding model to: {selected_model}[/green]")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error selecting embedding model: {e}")
            self.console.print(f"[red]Error selecting embedding model: {e}[/red]")
            return False 

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available embedding models with their details"""
        models = []
        for name, info in self.EMBEDDING_MODELS.items():
            model_type = "OpenAI API" if info['type'] == 'openai' else "Local (Hugging Face)"
            models.append({
                'name': name,
                'dimensions': info['dimensions'],
                'description': info['description'],
                'type': model_type
            })
        return models

    def unload_store(self) -> None:
        """Unload the current store and clean up resources"""
        try:
            if self.vectorstore:
                try:
                    # Persist any pending changes
                    if hasattr(self.vectorstore, '_persist'):
                        self.vectorstore._persist()
                    
                    # Get the underlying client if it exists
                    if hasattr(self.vectorstore, '_client'):
                        try:
                            # Try to stop the system client
                            if hasattr(self.vectorstore._client, '_system'):
                                self.vectorstore._client._system.stop()
                            # Clean up shared system client if using one
                            from chromadb.api.client import SharedSystemClient
                            if hasattr(self.vectorstore._client, '_identifier'):
                                SharedSystemClient._identifer_to_system.pop(
                                    self.vectorstore._client._identifier, None
                                )
                        except Exception as e:
                            self.logger.warning(f"Non-critical error during client cleanup: {e}")
                    
                    # Delete references to cleanup resources
                    self.vectorstore = None
                    self.store_name = None
                    self.current_directory = None
                    
                    # Update settings to reflect no store selected
                    settings = self._load_settings()
                    if 'agent' not in settings:
                        settings['agent'] = {}
                    settings['agent']['last_store'] = None
                    self._save_settings(settings)
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    self.logger.info("Successfully unloaded store")
                    self.console.print("[green]Successfully unloaded store[/green]")
                except Exception as e:
                    self.logger.error(f"Error during store cleanup: {e}")
                    self.console.print(f"[yellow]Warning: Store unloaded but some cleanup failed: {str(e)}[/yellow]")
                finally:
                    # Ensure references are cleared even if cleanup fails
                    self.vectorstore = None
                    self.store_name = None
                    self.current_directory = None
            else:
                self.logger.info("No store currently loaded")
        except Exception as e:
            self.logger.error(f"Error unloading store: {e}")
            self.console.print(f"[red]Error unloading store: {str(e)}[/red]") 