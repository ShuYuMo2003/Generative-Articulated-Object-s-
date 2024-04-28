from rich.progress import Progress
import time


with Progress(transient=True) as progress:
    task1 = progress.add_task("Downloading", total=100)
    task2 = progress.add_task("Processing", total=100)
    progress.start()
    for i in range(100):
        progress.update(task1, advance=1)
        progress.update(task2, advance=1)