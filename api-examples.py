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

#####################################################
# this does not work anymore, the api has changed
modelfile= '''
FROM llama3.2:latest
SYSTEM you are a cat, you answer only by meowing, purring and hissing
PARAMETER temperature 0.1
'''
#####################################################

res = ollama.create(
        model='cat', 
        from_='llama3.2',
        system='you are a cat, you answer only by meowing, purring and hissing',
        parameters={'temperature':0.1}
)
print(res)

res = ollama.generate(model='cat', prompt='hello')
print(res.response)

ollama.delete('cat')
models = ollama.list()
print([model.model for model in models.models])
stop_ollama()
