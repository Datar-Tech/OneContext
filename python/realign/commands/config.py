"""ReAlign config command."""

import typer
from pathlib import Path
from dataclasses import fields
from typing import Optional
from rich.console import Console
from rich.table import Table

from ..config import ReAlignConfig, get_default_config_content

console = Console()


def config_command(
    action: str = typer.Argument(..., help="Action to perform: 'get', 'set', or 'init'"),
    key: Optional[str] = typer.Argument(None, help="Configuration key (for get/set operations)"),
    value: Optional[str] = typer.Argument(None, help="Configuration value (for set operation)"),
):
    """
    Manage ReAlign configuration.

    Examples:

        aline config init                                    # Create default config file
        aline config get                                     # Show all config values
        aline config get use_LLM                             # Get specific config value
        aline config set llm_provider claude                 # Set LLM provider to Claude
        aline config set llm_provider openai                 # Set LLM provider to OpenAI
        aline config set llm_provider auto                   # Set LLM provider to auto
        aline config set anthropic_api_key sk-ant-xxx        # Set Anthropic API key
        aline config set openai_api_key sk-xxx               # Set OpenAI API key
    """
    config_path = Path.home() / ".aline" / "config.yaml"
    available_keys = [f.name for f in fields(ReAlignConfig)]

    if action == "init":
        # Initialize config file
        if config_path.exists():
            console.print(f"[yellow]Config file already exists at {config_path}[/yellow]")
            console.print("Use 'aline config get' to view current settings")
            return

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(get_default_config_content())

        console.print(f"[green]✓[/green] Created config file at {config_path}")
        console.print("\nEdit this file to customize your ReAlign settings")
        console.print("Set API keys via config commands:")
        console.print("  aline config set anthropic_api_key 'your-key'")
        console.print("  aline config set openai_api_key 'your-key'")
        return

    elif action == "get":
        # Get config value(s)
        if not config_path.exists():
            console.print(f"[red]✗[/red] Config file not found at {config_path}")
            console.print("Run 'aline config init' to create it")
            raise typer.Exit(1)

        config = ReAlignConfig.load()

        if key is None:
            # Show all config values
            table = Table(title="ReAlign Configuration", show_header=True, header_style="bold cyan")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            for field in fields(ReAlignConfig):
                field_value = getattr(config, field.name)
                # Mask API keys for security
                if "api_key" in field.name and field_value:
                    if len(field_value) > 8:
                        field_value = field_value[:4] + "..." + field_value[-4:]
                    else:
                        field_value = "***"
                table.add_row(field.name, str(field_value))

            console.print(table)
            console.print(f"\nConfig file location: {config_path}")
        else:
            # Show specific config value
            if not hasattr(config, key):
                console.print(f"[red]✗[/red] Unknown config key: {key}")
                console.print(f"Available keys: {', '.join(available_keys)}")
                raise typer.Exit(1)

            value = getattr(config, key)
            # Mask API keys for security
            if "api_key" in key and value:
                if len(value) > 8:
                    value = value[:4] + "..." + value[-4:]
                else:
                    value = "***"
            console.print(f"{key} = {value}")
        return

    elif action == "set":
        # Set config value
        if key is None or value is None:
            console.print("[red]✗[/red] Both key and value are required for 'set' action")
            console.print("Usage: aline config set <key> <value>")
            raise typer.Exit(1)

        if not config_path.exists():
            console.print(f"[yellow]⚠[/yellow] Config file not found, creating default config...")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(get_default_config_content())

        config = ReAlignConfig.load()

        # Validate and set the value
        if not hasattr(config, key):
            console.print(f"[red]✗[/red] Unknown config key: {key}")
            console.print(f"Available keys: {', '.join(available_keys)}")
            raise typer.Exit(1)

        # Type conversion and validation
        try:
            if key == "llm_provider":
                if value not in ("auto", "claude", "openai"):
                    console.print(f"[red]✗[/red] Invalid llm_provider value: {value}")
                    console.print("Valid values: auto, claude, openai")
                    raise typer.Exit(1)
            else:
                field_type = ReAlignConfig.__annotations__.get(key)
                if field_type is int:
                    value = int(value)
                elif field_type is bool:
                    value = value.lower() in ("true", "1", "yes")

            setattr(config, key, value)
            config.save()
            console.print(f"[green]✓[/green] Set {key} = {value}")

            if key == "llm_provider":
                console.print("\n[cyan]LLM Provider Options:[/cyan]")
                console.print("  auto   - Try Claude first, then OpenAI (default)")
                console.print("  claude - Use only Claude (Anthropic API)")
                console.print("  openai - Use only OpenAI (GPT API)")
                console.print("\nMake sure to set the appropriate API key:")
                if value in ("auto", "claude"):
                    console.print("  aline config set anthropic_api_key 'your-key'")
                if value in ("auto", "openai"):
                    console.print("  aline config set openai_api_key 'your-key'")

        except ValueError as e:
            console.print(f"[red]✗[/red] Invalid value for {key}: {e}")
            raise typer.Exit(1)

        return

    else:
        console.print(f"[red]✗[/red] Unknown action: {action}")
        console.print("Valid actions: get, set, init")
        raise typer.Exit(1)
