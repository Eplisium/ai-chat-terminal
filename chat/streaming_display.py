"""Streaming display module for real-time AI response visualization"""
import time
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.console import Group


class StreamingDisplay:
    """Handles real-time streaming display with Rich Live"""
    
    def __init__(self, console, model_name, instruction_name, colors):
        self.console = console
        self.model_name = model_name
        self.instruction_name = instruction_name
        self.colors = colors
        self.content = ""
        self.reasoning_content = ""
        self.tool_calls = []
        self.current_tool = None
        self.phase = "thinking"  # rag_search, thinking, reasoning, tool_call, responding
        self.start_time = time.time()
        self.live = None
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_index = 0
        self.rag_status = None  # Track RAG search status
        self.rag_results_count = 0  # Number of RAG results found
        
    def _get_spinner(self):
        """Get current spinner frame"""
        frame = self.spinner_frames[self.frame_index % len(self.spinner_frames)]
        self.frame_index += 1
        return frame
    
    def _get_elapsed_time(self):
        """Get formatted elapsed time"""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"
    
    def _build_display(self):
        """Build the current display renderable"""
        elements = []
        
        # Header with model info and elapsed time
        elapsed = self._get_elapsed_time()
        spinner = self._get_spinner()
        
        # RAG search section (if active or completed)
        if self.rag_status:
            rag_text = Text()
            if self.rag_status == "searching":
                rag_text.append(f"{spinner} ", style="bold magenta")
                rag_text.append("Searching knowledge base...", style="bold magenta")
            elif self.rag_status == "found":
                rag_text.append("✓ ", style="bold green")
                rag_text.append(f"Found {self.rag_results_count} relevant document(s)", style="green")
            elif self.rag_status == "none":
                rag_text.append("○ ", style="dim")
                rag_text.append("No relevant context found", style="dim")
            
            rag_panel = Panel(
                rag_text,
                title="[bold magenta]📚 RAG Context[/]",
                border_style="magenta",
                padding=(0, 1),
            )
            elements.append(rag_panel)
        
        # Reasoning section (if present)
        if self.reasoning_content:
            reasoning_text = Text()
            reasoning_text.append("💭 ", style="bold yellow")
            reasoning_text.append("Thinking...\n", style="bold yellow")
            # Show last 500 chars of reasoning to keep it compact
            display_reasoning = self.reasoning_content
            if len(display_reasoning) > 500:
                display_reasoning = "..." + display_reasoning[-497:]
            reasoning_text.append(display_reasoning, style="dim italic")
            
            reasoning_panel = Panel(
                reasoning_text,
                title="[bold yellow]🧠 Reasoning[/]",
                border_style="yellow",
                padding=(0, 1),
            )
            elements.append(reasoning_panel)
        
        # Tool calls section (if any)
        if self.tool_calls or self.current_tool:
            tool_content = Text()
            
            # Show completed tool calls
            for tool in self.tool_calls:
                tool_content.append("✓ ", style="bold green")
                tool_content.append(f"{tool['name']}", style="bold cyan")
                if tool.get('result_preview'):
                    tool_content.append(f" → {tool['result_preview'][:80]}...\n", style="dim")
                else:
                    tool_content.append("\n")
            
            # Show current tool being called
            if self.current_tool:
                tool_content.append(f"{spinner} ", style="bold yellow")
                tool_content.append(f"{self.current_tool['name']}", style="bold cyan")
                if self.current_tool.get('arguments'):
                    args_preview = str(self.current_tool['arguments'])[:60]
                    tool_content.append(f"({args_preview}...)", style="dim")
                tool_content.append("\n")
            
            if tool_content.plain:
                tool_panel = Panel(
                    tool_content,
                    title="[bold cyan]🔧 Tool Calls[/]",
                    border_style="cyan",
                    padding=(0, 1),
                )
                elements.append(tool_panel)
        
        # Main content section
        if self.content:
            # Show streaming content with cursor
            display_content = self.content
            if self.phase == "responding":
                display_content += "▌"
            
            content_panel = Panel(
                Markdown(display_content) if len(display_content) > 50 else Text(display_content),
                title=f"[bold {self.colors['ai_name']}]{self.model_name}[/] [bold {self.colors['instruction_name']}][{self.instruction_name}][/]",
                subtitle=f"[dim]{spinner} Streaming... ({elapsed})[/]",
                border_style="bright_blue",
                padding=(1, 2),
            )
            elements.append(content_panel)
        elif self.phase == "thinking" and not self.reasoning_content and not self.tool_calls:
            # Show initial thinking state
            thinking_text = Text()
            thinking_text.append(f"{spinner} ", style="bold yellow")
            thinking_text.append(f"{self.model_name} is thinking... ", style="bold yellow")
            thinking_text.append(f"({elapsed})", style="dim")
            
            thinking_panel = Panel(
                thinking_text,
                border_style="yellow",
                padding=(0, 1),
            )
            elements.append(thinking_panel)
        
        if elements:
            return Group(*elements)
        return Text(f"{spinner} Waiting for response...", style="bold yellow")
    
    def start(self):
        """Start the live display"""
        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self.live.start()
    
    def stop(self):
        """Stop the live display"""
        if self.live:
            # Clear the display before stopping to prevent ghost frames
            self.live.update("")
            self.live.stop()
            self.live = None
    
    def update(self):
        """Update the live display"""
        if self.live:
            self.live.update(self._build_display())
    
    def add_content(self, text):
        """Add content text"""
        self.content += text
        self.phase = "responding"
        self.update()
    
    def add_reasoning(self, text):
        """Add reasoning/thinking text"""
        self.reasoning_content += text
        self.phase = "reasoning"
        self.update()
    
    def start_tool_call(self, name, arguments=None):
        """Start a new tool call"""
        self.current_tool = {"name": name, "arguments": arguments}
        self.phase = "tool_call"
        self.update()
    
    def complete_tool_call(self, result_preview=None):
        """Complete the current tool call"""
        if self.current_tool:
            self.current_tool["result_preview"] = result_preview
            self.tool_calls.append(self.current_tool)
            self.current_tool = None
        self.update()
    
    def set_phase(self, phase):
        """Set the current phase"""
        self.phase = phase
        self.update()
    
    def start_rag_search(self):
        """Start RAG search indicator"""
        self.rag_status = "searching"
        self.phase = "rag_search"
        self.update()
    
    def complete_rag_search(self, results_count):
        """Complete RAG search with results count"""
        if results_count > 0:
            self.rag_status = "found"
            self.rag_results_count = results_count
        else:
            self.rag_status = "none"
        self.phase = "thinking"
        self.update()

