from imports import *
import sqlite3
from datetime import datetime, timedelta
from rich.table import Table
from rich.text import Text

class StatsManager:
    """Class to manage ACT statistics"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.db_path = os.path.join(self.base_dir, 'act_stats.db')
        self._init_db()

    def _init_db(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create model usage table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_cost REAL DEFAULT 0
                    )
                ''')
                
                # Create chat statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        token_count INTEGER,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        cost REAL DEFAULT 0
                    )
                ''')
                
                # Create system instruction usage table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS instruction_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        instruction_name TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            self.console.print("[red]Error initializing statistics database[/red]")

    def record_model_usage(self, model_config, total_cost=0):
        """Record when a model is used"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_usage (model_id, model_name, provider, total_cost)
                    VALUES (?, ?, ?, ?)
                ''', (model_config['id'], model_config['name'], model_config.get('provider', 'unknown'), total_cost))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error recording model usage: {e}")

    def record_chat(self, model_id, message_type, token_count=None, prompt_tokens=None, completion_tokens=None, cost=0):
        """Record chat statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_stats (model_id, message_type, token_count, prompt_tokens, completion_tokens, cost)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (model_id, message_type, token_count, prompt_tokens, completion_tokens, cost))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error recording chat stats: {e}")

    def record_instruction_usage(self, instruction_name):
        """Record system instruction usage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO instruction_usage (instruction_name)
                    VALUES (?)
                ''', (instruction_name,))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error recording instruction usage: {e}")

    def get_most_used_model(self):
        """Get the most frequently used model with total cost"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        model_name, 
                        provider, 
                        COUNT(*) as usage_count,
                        COALESCE(SUM(total_cost), 0) as total_cost
                    FROM model_usage
                    GROUP BY model_id
                    ORDER BY usage_count DESC
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                if result:
                    return {
                        'model_name': result[0],
                        'provider': result[1],
                        'usage_count': result[2],
                        'total_cost': result[3]
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting most used model: {e}")
            return None

    def get_chat_stats(self):
        """Get detailed chat statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total messages sent and received with token counts
                cursor.execute('''
                    SELECT 
                        message_type, 
                        COUNT(*) as count,
                        COALESCE(SUM(token_count), 0) as total_tokens,
                        COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                        COALESCE(SUM(cost), 0) as total_cost
                    FROM chat_stats
                    GROUP BY message_type
                ''')
                results = cursor.fetchall()
                stats = {}
                
                for result in results:
                    msg_type = result[0]
                    stats[msg_type] = {
                        'count': result[1],
                        'total_tokens': result[2],
                        'prompt_tokens': result[3],
                        'completion_tokens': result[4],
                        'cost': result[5]
                    }
                
                # Get total tokens and costs
                cursor.execute('''
                    SELECT 
                        COALESCE(SUM(token_count), 0) as total_tokens,
                        COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                        COALESCE(SUM(cost), 0) as total_cost
                    FROM chat_stats
                ''')
                totals = cursor.fetchone()
                
                return {
                    'sent': stats.get('sent', {'count': 0, 'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0}),
                    'received': stats.get('received', {'count': 0, 'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0}),
                    'totals': {
                        'total_tokens': totals[0],
                        'prompt_tokens': totals[1],
                        'completion_tokens': totals[2],
                        'total_cost': totals[3]
                    }
                }
        except Exception as e:
            self.logger.error(f"Error getting chat stats: {e}")
            return None

    def get_last_used_model(self):
        """Get the last used model with cost"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT model_name, provider, timestamp, COALESCE(total_cost, 0) as cost
                    FROM model_usage
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                if result:
                    # Format timestamp using current system time
                    try:
                        formatted_time = datetime.now().strftime('%Y-%m-%d %I:%M %p')
                    except:
                        formatted_time = result[2]
                    
                    return {
                        'model_name': result[0],
                        'provider': result[1],
                        'timestamp': formatted_time,
                        'cost': result[3]
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting last used model: {e}")
            return None

    def get_favorite_instruction(self):
        """Get the most frequently used system instruction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT instruction_name, COUNT(*) as usage_count
                    FROM instruction_usage
                    GROUP BY instruction_name
                    ORDER BY usage_count DESC
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                if result:
                    return {
                        'name': result[0],
                        'usage_count': result[1]
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting favorite instruction: {e}")
            return None

    def get_provider_stats(self):
        """Get statistics by provider"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        provider,
                        COUNT(*) as usage_count,
                        COALESCE(SUM(total_cost), 0) as total_cost
                    FROM model_usage
                    GROUP BY provider
                    ORDER BY usage_count DESC
                ''')
                results = cursor.fetchall()
                if results:
                    return [{
                        'provider': result[0],
                        'usage_count': result[1],
                        'total_cost': result[2]
                    } for result in results]
                return None
        except Exception as e:
            self.logger.error(f"Error getting provider stats: {e}")
            return None

    def get_daily_usage(self, days=7):
        """Get daily usage statistics for the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get dates for the last N days using system time
                dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days)]
                
                # Query for each date
                stats = []
                for date in dates:
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as message_count,
                            COALESCE(SUM(token_count), 0) as total_tokens,
                            COALESCE(SUM(cost), 0) as total_cost
                        FROM chat_stats
                        WHERE date(timestamp) = ?
                    ''', (date,))
                    result = cursor.fetchone()
                    if result[0] > 0:  # Only include days with activity
                        stats.append({
                            'date': date,
                            'message_count': result[0],
                            'total_tokens': result[1],
                            'total_cost': result[2]
                        })
                
                return stats if stats else None
        except Exception as e:
            self.logger.error(f"Error getting daily usage: {e}")
            return None

    def format_stats_display(self):
        """Format statistics for display using Rich tables"""
        # Get all statistics
        most_used = self.get_most_used_model()
        last_used = self.get_last_used_model()
        chat_stats = self.get_chat_stats()
        fav_instruction = self.get_favorite_instruction()
        provider_stats = self.get_provider_stats()
        daily_usage = self.get_daily_usage(7)  # Last 7 days

        # Create main stats table
        main_table = Table(
            title="📊 ACT Statistics",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            padding=(0, 1)
        )
        main_table.add_column("Category", style="cyan", no_wrap=True)
        main_table.add_column("Details", style="white")

        # Most Used Model
        if most_used:
            main_table.add_row(
                "Most Used Model",
                Text.from_markup(
                    f"[bold]{most_used['model_name']}[/] ([cyan]{most_used['provider']}[/])\n"
                    f"Used [green]{most_used['usage_count']}[/] times\n"
                    f"Total Cost: [yellow]${most_used['total_cost']:.6f}[/]"
                )
            )
        else:
            main_table.add_row("Most Used Model", "No data available")

        # Last Used Model
        if last_used:
            main_table.add_row(
                "Last Used Model",
                Text.from_markup(
                    f"[bold]{last_used['model_name']}[/] ([cyan]{last_used['provider']}[/])\n"
                    f"Last used: [green]{last_used['timestamp']}[/]\n"
                    f"Cost: [yellow]${last_used['cost']:.6f}[/]"
                )
            )
        else:
            main_table.add_row("Last Used Model", "No data available")

        # Chat Statistics
        if chat_stats:
            sent = chat_stats['sent']
            received = chat_stats['received']
            totals = chat_stats['totals']
            main_table.add_row(
                "Chat Statistics",
                Text.from_markup(
                    f"Messages Sent: [green]{sent['count']:,}[/]\n"
                    f"Messages Received: [green]{received['count']:,}[/]\n"
                    f"Total Tokens: [cyan]{totals['total_tokens']:,}[/]\n"
                    f"• Prompt Tokens: [blue]{totals['prompt_tokens']:,}[/]\n"
                    f"• Completion Tokens: [blue]{totals['completion_tokens']:,}[/]\n"
                    f"Total Cost: [yellow]${totals['total_cost']:.6f}[/]"
                )
            )
        else:
            main_table.add_row("Chat Statistics", "No data available")

        # Favorite System Instruction
        if fav_instruction:
            main_table.add_row(
                "Favorite Instruction",
                Text.from_markup(
                    f"[bold]{fav_instruction['name']}[/]\n"
                    f"Used [green]{fav_instruction['usage_count']}[/] times"
                )
            )
        else:
            main_table.add_row("Favorite Instruction", "No data available")

        # Provider Statistics
        if provider_stats:
            provider_details = []
            for stat in provider_stats:
                provider_details.append(
                    f"[cyan]{stat['provider'].upper()}[/]: "
                    f"[green]{stat['usage_count']}[/] uses "
                    f"([yellow]${stat['total_cost']:.6f}[/])"
                )
            main_table.add_row("Provider Usage", Text.from_markup("\n".join(provider_details)))
        else:
            main_table.add_row("Provider Usage", "No data available")

        # Daily Usage
        if daily_usage:
            daily_details = []
            for day in daily_usage:
                daily_details.append(
                    f"[cyan]{day['date']}[/]: "
                    f"[green]{day['message_count']}[/] msgs, "
                    f"[blue]{day['total_tokens']:,}[/] tokens, "
                    f"[yellow]${day['total_cost']:.6f}[/]"
                )
            main_table.add_row("Last 7 Days", Text.from_markup("\n".join(daily_details)))
        else:
            main_table.add_row("Last 7 Days", "No data available")

        return main_table 