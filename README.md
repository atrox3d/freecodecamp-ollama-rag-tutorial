
# FreeCodeCamp Ollama Tutorial

## Description
This project is a FreeCodeCamp tutorial for using Ollama. It provides step-by-step instructions and examples to help you get started.

## System dependencies:

langchain requires the following system dependencies:

    - poppler
    - tesseract
    - (tesseract-lang)

## Extra dependencies:

to avoid LookupError:

    import nltk
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')


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


## Using basic.py

basic.py sends a prompt to ollama server using only requests:
- starts ollama server
- sends the prompt
- returns the answer
- stops the server


```bash
 Usage: basic.py [OPTIONS] PROMPT                                                   
                                                                                    
 typer interface to chat function                                                   
                                                                                    
╭─ Arguments ──────────────────────────────────────────────────────────────────────╮
│ *    prompt      TEXT  [default: None] [required]                                │
╰──────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────╮
│ --host                     TEXT     [default: localhost]                         │
│ --port                     INTEGER  [default: 11434]                             │
│ --endpoint                 TEXT     [default: api/generate]                      │
│ --model                    TEXT     [default: llama3.2]                          │
│ --wait                     FLOAT    [default: 0.1]                               │
│ --attempts                 INTEGER  [default: 10]                                │
│ --stop        --no-stop             [default: stop]                              │
│ --help                              Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

## Using pdf-rag.py

pdf-rag.py loads a pdf into a vector-db and sends a question about it to the llm:
- starts ollama server
- loads pdf
- splits and vectorize pdf
- creates chain
- sends the prompt with the question
- returns the answer
- stops the server


```bash
Usage: pdf-rag.py [OPTIONS] QUESTION                                               
                                                                                    
 typer command representig main                                                     
                                                                                    
╭─ Arguments ──────────────────────────────────────────────────────────────────────╮
│ *    question      TEXT  [default: None] [required]                              │
╰──────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────╮
│ --doc-path                   TEXT     [default: data/BOI.pdf]                    │
│ --model                      TEXT     [default: llama3.2]                        │
│ --embeddings                 TEXT     [default: nomic-embed-text]                │
│ --host                       TEXT     [default: localhost]                       │
│ --port                       INTEGER  [default: 11434]                           │
│ --wait                       FLOAT    [default: 0.1]                             │
│ --attempts                   INTEGER  [default: 10]                              │
│ --stop          --no-stop             [default: stop]                            │
│ --help                                Show this message and exit.                │
╰──────────────────────────────────────────────────────────────────────────────────╯
```
