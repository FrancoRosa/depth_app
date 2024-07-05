import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import subprocess

class ChangeHandler(FileSystemEventHandler):
    """Restart the application when the code changes."""
    def on_modified(self, event):
        if event.src_path.endswith("your_script_name.py"):
            print(f'Changes detected in {event.src_path}. Restarting app...')
            os.execv(__file__, ['python'] + sys.argv)

def start_observer(path):
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    path = '/home/athena/Documents/GitHub/OAKD-Depth/tkinter-app'
    start_observer(path)
    subprocess.call(['python', 'app.py'])
