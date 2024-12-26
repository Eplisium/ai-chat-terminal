from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import errno
from dotenv import load_dotenv
import json
from rich.console import Console
from rich.panel import Panel
import logging
import inquirer
import glob
import datetime
import time

class FileContent(BaseModel):
    """Model for file content"""
    filename: str = Field(..., description="Name of the file")
    content: str = Field(..., description="Content of the file")
    file_type: str = Field(..., description="Type/extension of the file")

class ChromaManager:
    # Available embedding models and their dimensions
    EMBEDDING_MODELS = {
        'text-embedding-3-small': {'dimensions': 1536, 'description': 'Smallest and most cost-effective model'},
        'text-embedding-3-large': {'dimensions': 3072, 'description': 'Most capable model for complex tasks'},
        'text-embedding-ada-002': {'dimensions': 1536, 'description': 'Legacy model, good balance of performance and cost'}
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
        
        # Create stores directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize embeddings
        self.initialize_embeddings()
    
    def initialize_embeddings(self) -> None:
        """Initialize OpenAI embeddings with the selected model"""
        try:
            # Load environment variables again to ensure they're available
            load_dotenv()
            
            # Get API key from environment
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

            # Load settings to get the selected embedding model
            settings = self._load_settings()
            selected_model = settings.get('chromadb', {}).get('embedding_model', 'text-embedding-3-small')

            # Initialize embeddings with explicit parameters
            self.embeddings = OpenAIEmbeddings(
                model=selected_model,
                openai_api_key=self.openai_api_key,
                chunk_size=1000,  # Process texts in smaller chunks
                max_retries=3     # Retry failed requests
            )
            
            self.logger.info(f"Successfully initialized OpenAI embeddings with model: {selected_model}")
            self.console.print(f"[green]Successfully initialized embeddings with model: {selected_model}[/green]")
                
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = None
            self.console.print(Panel(
                f"[red]Error initializing embeddings:[/red]\n{str(e)}\n\n"
                "[yellow]Please check:[/yellow]\n"
                "1. Your OpenAI API key is valid\n"
                "2. You have access to the selected embedding model\n"
                "3. Your internet connection is working",
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

            self.console.print("[cyan]Testing OpenAI embeddings...[/cyan]")
            _ = self.embeddings.embed_query("Test query")
            self.console.print("[green]Embeddings test successful![/green]")
            return True
        except Exception as e:
            self.logger.error(f"Failed to test embeddings: {e}")
            self.embeddings = None
            self.console.print(f"[red]Embeddings test failed: {str(e)}[/red]")
            return False

    def refresh_store(self, directory_path: str = None, files: List[str] = None) -> bool:
        """Refresh the current store with the same directory or specific files"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False

            # Get current store name and metadata before recreating
            current_store_name = self.store_name
            store_path = os.path.join(self.persist_directory, current_store_name)
            metadata_path = os.path.join(store_path, 'metadata.json')
            last_directory = None

            # Try to get the last directory from metadata
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        last_directory = metadata.get('last_directory')
            except Exception as e:
                self.logger.error(f"Error reading metadata: {e}")

            # Use provided directory, files, or last directory
            target_directory = directory_path or last_directory
            if not target_directory and not files:
                self.console.print("[yellow]No directory or files specified for refresh[/yellow]")
                return False

            # Unload current store
            self.unload_store()

            # Delete the existing store
            try:
                store_path = os.path.join(self.persist_directory, current_store_name)
                if os.path.exists(store_path):
                    import shutil
                    def handle_remove_readonly(func, path, exc_info):
                        """Handle read-only files during deletion"""
                        import stat
                        if func in (os.unlink, os.rmdir) and exc_info[1].errno == errno.EACCES:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                    shutil.rmtree(store_path, onerror=handle_remove_readonly)
            except Exception as e:
                self.logger.error(f"Error deleting store directory: {e}")
                return False

            # Create a new store with the same name
            if not self.create_store(current_store_name):
                self.logger.error("Failed to recreate store")
                return False

            # Process the directory or files
            if target_directory:
                self.console.print(f"[cyan]Using directory: {target_directory}[/cyan]")
                return self.process_directory(target_directory, force_refresh=True)
            elif files:
                # Read and process specific files
                processed_files = []
                for filepath in files:
                    try:
                        if os.path.exists(filepath):
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                filename = os.path.basename(filepath)
                                processed_files.append(FileContent(
                                    filename=filename,
                                    content=content,
                                    file_type=os.path.splitext(filename)[1]
                                ))
                    except Exception as e:
                        self.logger.error(f"Error reading {filepath}: {str(e)}")

                if processed_files:
                    # Create documents for ChromaDB
                    self.console.print("[cyan]Creating embeddings for updated files...[/cyan]")
                    documents = [
                        f"File: {file.filename}\nContent:\n{file.content}"
                        for file in processed_files
                    ]
                    
                    # Store in ChromaDB
                    self.vectorstore.add_texts(documents)
                    self.console.print("[green]Successfully refreshed files in ChromaDB store[/green]")
                    return True

            return False
            
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

    def read_directory(self, directory_path: str, force_refresh: bool = False) -> List[FileContent]:
        """Read all files in the directory matching the configured patterns"""
        try:
            settings = self._load_settings()
            chromadb_settings = settings.get('chromadb', {})
            file_types = chromadb_settings.get('file_types', [])
            exclude_patterns = chromadb_settings.get('exclude_patterns', [])
            max_size_mb = chromadb_settings.get('max_file_size_mb', 5)
            
            files = []
            self.console.print("[cyan]Reading files...")
            
            # Create glob patterns for each file type
            for file_type in file_types:
                pattern = os.path.join(directory_path, f"**/*{file_type}")
                for filepath in glob.glob(pattern, recursive=True):
                    try:
                        # Check if file should be excluded
                        if any(exclude in filepath for exclude in exclude_patterns):
                            continue
                        
                        # Check file size
                        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        if file_size_mb > max_size_mb:
                            self.logger.warning(f"Skipping {filepath}: File too large ({file_size_mb:.1f}MB > {max_size_mb}MB)")
                            continue
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            filename = os.path.basename(filepath)
                            file_type = os.path.splitext(filename)[1]
                            files.append(FileContent(
                                filename=filename,
                                content=content,
                                file_type=file_type
                            ))
                            self.console.print(f"[cyan]Read: {filename}")
                    except Exception as e:
                        self.logger.error(f"Error reading {filepath}: {str(e)}")
            
            return files
            
        except Exception as e:
            self.logger.error(f"Error reading directory: {str(e)}")
            return []

    def process_directory(self, directory_path: str, force_refresh: bool = False) -> bool:
        """Process all files in a directory and add them to the current store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            # Save the directory path for future refreshes
            if self.store_name:
                store_path = os.path.join(self.persist_directory, self.store_name)
                metadata_path = os.path.join(store_path, 'metadata.json')
                try:
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}
                    
                    metadata['last_directory'] = os.path.abspath(directory_path)
                    metadata['last_processed'] = datetime.datetime.now().isoformat()
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                except Exception as e:
                    self.logger.error(f"Error updating metadata: {e}")
            
            # Only read and embed files if vectorstore is empty or force refresh is True
            if force_refresh or self.vectorstore._collection.count() == 0:
                self.console.print("[cyan]Reading directory contents...")
                files = self.read_directory(directory_path)
                
                if not files:
                    self.console.print("[yellow]No valid files found to process[/yellow]")
                    return False
                
                # Create documents for ChromaDB
                self.console.print("[cyan]Creating embeddings...")
                documents = [
                    f"File: {file.filename}\nContent:\n{file.content}"
                    for file in files
                ]
                
                # Store in ChromaDB
                self.vectorstore.add_texts(documents)
                self.console.print("[green]Successfully added files to ChromaDB store[/green]")
            
            return True
            
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
                'embedding_dimensions': self.EMBEDDING_MODELS[current_model]['dimensions']
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
    
    def add_file_to_store(self, filepath: str, content: str) -> bool:
        """Add a file's content to the current store"""
        try:
            if not self.vectorstore:
                self.logger.error("No store currently loaded")
                return False
            
            document = f"File: {filepath}\nContent:\n{content}"
            self.vectorstore.add_texts([document])
            return True
        except Exception as e:
            self.logger.error(f"Error adding file to store: {e}")
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
                    message="Select OpenAI Embedding Model",
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

            self.console.print(f"[green]Successfully updated embedding model to: {selected_model}[/green]")
            return True

        except Exception as e:
            self.logger.error(f"Error selecting embedding model: {e}")
            self.console.print(f"[red]Error selecting embedding model: {e}[/red]")
            return False 

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