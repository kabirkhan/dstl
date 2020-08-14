import typer

from .translate import translate

app = typer.Typer(no_args_is_help=True)


commands = [translate]
for command in commands:
    app.command(no_args_is_help=True)(command)


@app.callback()
def main() -> None:
    """
    \b
      _     _                       _       _       
     | |   | |                     | |     | |      
   __| |___| |_ _ __ __ _ _ __  ___| | __ _| |_ ___ 
  / _` / __| __| '__/ _` | '_ \/ __| |/ _` | __/ _ |
 | (_| \__ \ |_| | | (_| | | | \__ \ | (_| | ||  __/
  \__,_|___/\__|_|  \__,_|_| |_|___/_|\__,_|\__\___|                                                    
    """
    pass


if __name__ == "__main__":
    app()
