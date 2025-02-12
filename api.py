import ollama

import defaults
from ollamaserver import start_ollama, stop_ollama, ollama_up


start_ollama()
# print installed models
response:ollama.ListResponse = ollama.list()
# print([model['model']for model in response.model_dump()['models']])
# print([model.model for model in response.models])


res = ollama.chat(
    model='llama3.2',
    messages=[
        {
            'role': 'user',
            'content': 'Hello, how are you?'
        },
    ],
    stream=True
)

# to use the old dict interface use model_dump()
# print(res.model_dump()['message']['content'])
# print(res.message.content)
for chunk in res:
    print(chunk.message.content, end='', flush=True)
print()


res = ollama.generate(
    model='llama3.2',
    prompt='Hello, how are you?',
    stream=True
)
for chunk in res:
    print(chunk.response, end='', flush=True)
print()

show = ollama.show('llama3.2')
print(show.modelinfo)

stop_ollama()
