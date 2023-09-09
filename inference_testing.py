from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# Storing lines in a list
lines = []

### Open the file and read its lines
with open("inputs.txt", "r") as file:
    for line in file:
        lines.append(line.strip())

### Initializing model and tokenizer

model = T5ForConditionalGeneration.from_pretrained("my-t5-hinglish-translator")  
tokenizer = T5Tokenizer.from_pretrained("my-t5-hinglish-translator")

output = {}

### Generating translation and saving it as a json file
for line in lines:
    input = tokenizer(f"translate English to Hinglish: {line}?", return_tensors="pt").input_ids
    output = model.generate(input)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    output[line] = decoded_output

with open("results.json", "w") as json_file:
    json.dump(output, json_file)

