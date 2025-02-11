
# FreeCodeCamp Ollama Tutorial

## Description
This project is a FreeCodeCamp tutorial for using Ollama. It provides step-by-step instructions and examples to help you get started.

## Creating a cat assistant

1. Create the model from file:
    ```bash
    ollama create brian -f ./Modelfile
    ```

2. Run the model:
    ```bash
    ollama run brian
    ```
    
3. Remove the model:
    ```bash
    ollama rm brian
    ```


## Use ollama server

### use the generate endpoint

1. Open ollama app, or in the terminal:
    ```bash
    ollama serve
    ```

2. Create a request with curl (with streaming enabled):
    ```bash
    curl http://localhost:11434/api/generate -d '{
        "model": "llama3.2",
        "prompt": "why the sky is blue?"
    }'
    ```

3. Create a request with curl (with streaming disabled):
    ```bash
    curl http://localhost:11434/api/generate -d '{
        "model": "llama3.2",
        "prompt": "why the sky is blue?",
        "stream": false
    }'
    ```

### use the chat endpoint

1. Create a request with curl (with streaming disabled):
    ```bash
    curl http://localhost:11434/api/chat -d '{
        "model": "llama3.2",
        "messages": [
            {
                "role"   : "user",
                "content": "do you know brian the cat llm?"
            }
        ],
        "stream": false
    }'
    ```

### Specify JSON output

1. Ask for output format and specify it in the payload:
    ```bash
    curl http://localhost:11434/api/generate -d '{
        "model": "llama3.2",
        "prompt": "what color is the sky at different times of the day? answer using JSON",
        "format": "json",
        "stream": false
    }'
    ```
