import requests
import json
import typer
import ollamamanager

import defaults
from ollamamanager import with_ollama_up


app = typer.Typer(add_completion=False)


@app.command()
def main(
    prompt  :str,                            # chat function parameter
    host    :str   = defaults.HOST,          # chat function parameter
    port    :int   = defaults.PORT,          # chat function parameter
    endpoint:str   = defaults.ENDPOINT,      # chat function parameter
    model   :str   = 'llama3.2',             # chat function parameter
    wait    :float = defaults.WAIT_SECONDS,  # decorator parameter
    attempts:int   = defaults.ATTEMPTS,      # decorator parameter
    stop    :bool  = True                    # decorator parameter
):
    '''typer interface to chat function'''
    print(f'{host = }')
    with ollamamanager.OllamaServerCtx(host, port, wait, attempts, stop):
        chat(
            prompt, host, port, endpoint, model,
            #
            # these params are consumed by the decorator but with remote ollama
            # things get complicated
            #
            # decorator_wait=wait, 
            # decorator_stop=stop,
            # decorator_attempts=attempts
        )


# @with_ollama_up()
def chat(
    prompt  :str,
    host    :str,
    port    :int,
    endpoint:str,
    model   :str,
):
    '''sends a request to ollama server'''
    data = {
        'model': model,
        'prompt': prompt,
    }

    print(f'{host = }')
    url = f'http://{host}:{port}/{endpoint}'
    
    try:
        response = requests.post(url, json=data, stream=True)

        if response.status_code == 200:
            print('Generated text:', end='', flush=True)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    result = json.loads(decoded_line)
                    generated_text = result.get('response', '')
                    print(generated_text, end='', flush=True)
            print()
        else:
            print('Error:', response.status_code)
    except Exception as ce:
        print(ce)

if __name__ == "__main__":
    app()