import subprocess
from time import sleep
import socket
import time

import defaults


def is_server_ready(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_server(host, port, wait):
    # Wait for the server to be ready
    while not is_server_ready(host, port):
        print("Waiting for the server to be ready...")
        time.sleep(wait)


def start_ollama(
    host: str = defaults.HOST, 
    port=11434, 
    wait=.1
):
    print('starting ollama server...')
    completed = subprocess.run('ollama serve > /dev/null 2>&1 &', shell=True, check=True)

    wait_for_server(host, port, wait)
    
    return completed


def stop_ollama():
    print('stopping ollama server...')
    completed = subprocess.run('pkill ollama', shell=True, check=True)
    return completed


def ollama_up(
    host:str   = defaults.HOST, 
    port:int   = defaults.PORT, 
    wait:float = defaults.WAIT_SECONDS, 
    stop:bool  = True
):
    def decorator(func):
        def wrap(*args, **kwargs):
            nonlocal wait                               # access enclosing parameter
            nonlocal stop                               # access enclosing parameter
            wait = kwargs.pop('decorator_wait', wait)   # use kwarg and remove it
            stop = kwargs.pop('decorator_stop', stop)   # use kwarg and remove it
            try:
                start_ollama(host, port, wait)
                ret = func(*args, **kwargs)
                return ret
            finally:
                if stop:
                    stop_ollama()
        return wrap
    return decorator


if __name__ == "__main__":
    print(start_ollama())
    print(stop_ollama())
