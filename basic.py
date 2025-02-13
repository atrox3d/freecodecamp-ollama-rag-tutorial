import requests
import json
import typer

import defaults
from ollamamanager import ollama_up


app = typer.Typer()


@app.command()
def main(
    prompt  :str,                            # chat function parameter
    host    :str   = defaults.HOST,          # chat function parameter
    port    :int   = defaults.PORT,          # chat function parameter
    endpoint:str   = defaults.ENDPOINT,      # chat function parameter
    model   :str   = 'llama3.2',             # chat function parameter
    wait    :float = defaults.WAIT_SECONDS,  # decorator parameter
    stop    :bool  = True                    # decorator parameter
):
    '''typer interface to chat function'''
    chat(
        prompt, host, port, endpoint, model, 
        decorator_wait=wait, 
        decorator_stop=stop
    )


@ollama_up()
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