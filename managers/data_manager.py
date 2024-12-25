from imports import *

class DataManager:
    """Class to manage application data operations"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.base_dir = os.path.dirname(os.path.dirname(__file__))

    def clear_logs(self):
        """Clear all log files from the logs directory"""
        try:
            logs_dir = os.path.join(self.base_dir, 'logs')
            if os.path.exists(logs_dir):
                cleared = False
                for file in os.listdir(logs_dir):
                    if file.endswith('.log'):
                        file_path = os.path.join(logs_dir, file)
                        try:
                            os.remove(file_path)
                            cleared = True
                        except Exception as e:
                            self.logger.error(f"Error deleting log file {file}: {e}")
                
                if cleared:
                    self.console.print("[green]Successfully cleared all log files[/green]")
                else:
                    self.console.print("[yellow]No log files found to clear[/yellow]")
            else:
                self.console.print("[yellow]No logs directory found[/yellow]")
        except Exception as e:
            self.logger.error(f"Error clearing logs: {e}")
            self.console.print("[red]Error clearing logs[/red]")

    def clear_chats(self):
        """Clear all chat history files and subdirectories from the chats directory"""
        try:
            chats_dir = os.path.join(self.base_dir, 'chats')
            if os.path.exists(chats_dir):
                cleared = False
                # First, walk through all directories and files
                for root, dirs, files in os.walk(chats_dir, topdown=False):
                    # Delete all files in current directory
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            cleared = True
                        except Exception as e:
                            self.logger.error(f"Error deleting chat file {file}: {e}")
                    
                    # Delete all subdirectories except the main chats directory
                    if root != chats_dir:
                        try:
                            os.rmdir(root)
                            cleared = True
                        except Exception as e:
                            self.logger.error(f"Error deleting directory {root}: {e}")
                
                if cleared:
                    self.console.print("[green]Successfully cleared all chat history and subdirectories[/green]")
                else:
                    self.console.print("[yellow]No chat files or directories found to clear[/yellow]")
            else:
                self.console.print("[yellow]No chats directory found[/yellow]")
        except Exception as e:
            self.logger.error(f"Error clearing chats: {e}")
            self.console.print("[red]Error clearing chat history[/red]") 