import sys
import time
import threading
from colorama import init as init_colorama, Fore
# Initialize colorama
init_colorama()

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None, description=None, color: str = Fore.RED):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay
        self.description = description
        self.color = color

    def spinner_task(self):
        start_time = time.time()
        while self.busy:
            elapsed_time = time.time() - start_time
            sys.stdout.write(self.color + f"\r{next(self.spinner_generator)} Running... Elapsed Time: {elapsed_time:.2f}s {'- ' + self.description if self.description else ''}" + Fore.RESET)
            sys.stdout.flush()
            time.sleep(self.delay)

    def __enter__(self):
        self.busy = True
        self.spinner_threading = threading.Thread(target=self.spinner_task)
        self.spinner_threading.start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        print()
        if exception is not None:
            return False

if __name__ == '__main__':
    print("Hi")
    with Spinner(description="Processing", color=Fore.BLUE):
        # ... some long-running operations
        time.sleep(5)
    with Spinner(description="Processing", color=Fore.BLUE):
        # ... some long-running operations
        time.sleep(5)
