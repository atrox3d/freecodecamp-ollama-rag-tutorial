import subprocess
from time import sleep
import socket
import time

import defaults


def is_server_ready(host, port):
    '''check if server ready'''
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_server(host, port, wait, max_attempts):
    '''Wait for the server to be ready'''
    server = f'{host}:{port}'
    attempts = 0
    while not is_server_ready(host, port):
        print("Waiting for the server to be ready...")
        time.sleep(wait)
        attempts += 1
        if attempts > max_attempts:
            raise TimeoutError(f'no answer from server {server} after {attempts} attempts')


def start_ollama(
    host: str = defaults.HOST, 
    port=11434, 
    wait=.1,
    attempts=10
):
    '''start ollama server and wait for it to be up'''
    print('starting ollama server...')
    completed = subprocess.run('ollama serve > /dev/null 2>&1 &', shell=True, check=True)

    wait_for_server(host, port, wait, attempts)
    
    return completed


def stop_ollama():
    '''stop ollama server'''
    print('stopping ollama server...')
    completed = subprocess.run('pkill ollama', shell=True, check=True)
    return completed


def with_ollama_up(
    host:str     = defaults.HOST, 
    port:int     = defaults.PORT, 
    wait:float   = defaults.WAIT_SECONDS,
    attempts:int = defaults.ATTEMPTS,
    stop:bool  = True
):
    '''parameterize decorator'''
    def decorator(func):
        '''ollama server up decorator'''
        def wrap(*args, **kwargs):
            '''enclose function with ollama up/down'''
            nonlocal wait                                           # access enclosing parameter
            nonlocal attempts                                       # access enclosing parameter
            nonlocal stop                                           # access enclosing parameter
            wait = kwargs.pop('decorator_wait', wait)               # use kwarg and remove it
            attempts = kwargs.pop('decorator_attempts', attempts)   # use kwarg and remove it
            stop = kwargs.pop('decorator_stop', stop)               # use kwarg and remove it
            try:
                start_ollama(host, port, wait, attempts)
                ret = func(*args, **kwargs)
                return ret
            finally:
                # we can leave ollama server up
                if stop:
                    stop_ollama()
        return wrap
    return decorator


def is_ollama_up(host:str=defaults.HOST, port:int=defaults.PORT) -> bool:
    return is_server_ready()


class OllamaCtx:
    """
    A simple context manager class that starts and stops ollama, taking parameters
    for the start and stop functions.
    """
    def __init__(self, 
            host: str    = defaults.HOST, 
            port: int    = defaults.PORT, 
            wait: float  = defaults.WAIT_SECONDS, 
            attempts:int = defaults.ATTEMPTS,
            stop:bool    = True
        ):
        """
        Initializes the ServerContext with the host and port.
        """
        self.host     = host
        self.port     = port
        self.wait     = wait
        self.attempts = attempts
        self.stop     = stop

    def __enter__(self):
        start_ollama(self.host, self.port, self.wait, self.attempts)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stop:
            stop_ollama()



if __name__ == "__main__":
    # test context manager
    with OllamaCtx():
        raise Exception()
