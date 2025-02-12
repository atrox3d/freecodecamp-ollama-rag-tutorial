import requests
import json
import typer

import defaults
from ollamahelper import ollamaserver_decorator

app = typer.Typer()


@app.command()
def main(
    prompt  :str,
    server  :str = defaults.HOST,
    port    :int = defaults.PORT,
    endpoint:str = defaults.ENDPOINT,
    model   :str = 'llama3.2',
):
    chat(prompt, server, port, endpoint, model)


@ollamaserver_decorator
def chat(
    prompt  :str,
    server  :str,
    port    :int,
    endpoint:str,
    model   :str,
):
    data = {
        'model': model,
        'prompt': prompt,
    }

    url = f'http://{server}:{port}/{endpoint}'
    
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