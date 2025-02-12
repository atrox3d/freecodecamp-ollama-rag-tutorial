import os
import ollama

import defaults
from ollamaserver import start_ollama, stop_ollama, ollama_up


start_ollama()

model = 'llama3.2'
input_file = 'data/grocery_list.txt'
output_file = 'data/categorized_grocery_list.txt'

if not os.path.exists(input_file):
    print(f'not found {input_file = }')
    exit(1)

with open(input_file) as f:
    items = f.read().strip()
    print(f'{items.split() }')

prompt = '''
you are an assistant that categorizes and sorts grocery items.

here is a list of grocery items:

{items}

Pleae:


1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.
4. do not output the initial message, only output the data
'''

try:
        response = ollama.generate(model=model, prompt=prompt)
        output = response.response
        
        with open(output_file, 'w') as f:
            f.write(output.lstrip())
        
except:
     pass

stop_ollama()