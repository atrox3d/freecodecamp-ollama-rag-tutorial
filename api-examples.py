import ollama

import defaults
from ollamamanager import start_ollama, stop_ollama


start_ollama()

######################################################################
# print installed models
######################################################################

response:ollama.ListResponse = ollama.list()
# print([model['model']for model in response.model_dump()['models']])
# print([model.model for model in response.models])


######################################################################
# create chat
######################################################################
# chat keeps a context and history
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

#*************************************************
# changed in ollama at least 0.4.7
#*************************************************
# to use the old dict interface use model_dump()
# print(res.model_dump()['message']['content'])
# print(res.message.content)
#*************************************************
for chunk in res:
    print(chunk.message.content, end='', flush=True)
print()


######################################################################
# create response
######################################################################
# generate does not keep a context
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

######################################################################
# create model
######################################################################

#*************************************************
# this does not work anymore, the api has changed
#*************************************************
modelfile= '''
FROM llama3.2:latest
SYSTEM you are a cat, you answer only by meowing, purring and hissing
PARAMETER temperature 0.1
'''
#*************************************************

# new create interface for version 0.4.7
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
