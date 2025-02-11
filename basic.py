import requests
import json
import typer


SERVER = 'localhost'
PORT = 11434
ENDPOINT = 'api/generate'
URL = f'http://{SERVER}:{PORT}/{ENDPOINT}'

app = typer.Typer()

@app.command()
def chat(
    prompt  :str,
    server  :str = SERVER,
    port    :int = PORT,
    endpoint:str = ENDPOINT,
    model   :str = 'llama3.2',
):
    data = {
        'model': model,
        'prompt': prompt,
    }

    url = f'http://{server}:{port}/{endpoint}'
    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        print('Generated text:', end='', flush=True)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                result = json.loads(decoded_line)
                generated_text = result.get('response', '')
                print(generated_text, end='', flush=True)
    else:
        print('Error:', response.status_code)


if __name__ == "__main__":
    app()