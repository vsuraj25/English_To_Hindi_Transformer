from transformers import (AutoConfig, AutoTokenizer,AutoModelForSeq2SeqLM, default_data_collator, HfArgumentParser,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from datasets import load_data

### MODEL ###
model_name = "t5-small"
config = AutoConfig.from_pretrained(
    model_name
  )
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True, #important
  )
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    config=config
  )

### Data Tokenizer
data = load_data('findnitai/english-to-hinglish')


master = []
for line in data['train']['translation']:
    master.append(line['en'])
    master.append(line['hi_ng'])

def gen_training_data():
    return (master[i : i+500]
    for i in range(0, len(master), 500)
    )
tokenizer_training_data = gen_training_data()
tokenizer = tokenizer.train_new_from_iterator(tokenizer_training_data, 32128)

## Loading and training data
train_file = "hinglish_upload_v1.json" 

data_files = {}
data_files["train"] = train_file

raw_data = load_data(
    "json",
    data_files=data_files
  )
source_prefix = "Translate English to Hinglish : "
source_lang = "en"
target_lang = "hi_ng"
max_source_length = 128
max_target_length = 128 
padding = "max_length" 
num_epochs = 5

def preprocess(source_data):
    inputs = [sample[source_lang] for sample in source_data["translation"]]
    targets = [sample[target_lang] for sample in source_data["translation"]]
    inputs = [source_prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
    # replace tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss.
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_data = raw_data["train"]
train_data = train_data.map(preprocess, batched=True, remove_columns="translation")
data_collator = default_data_collator


trainer_args_in = {
    'output_dir': 't5-hinglish-model',
    'overwrite_output_dir' : True,
    'do_train' : True,
    'per_device_train_batch_size' : 8,
    'num_train_epochs' : num_epochs,
}


parser = HfArgumentParser((Seq2SeqTrainingArguments))
training_args = parser.parse_dict(trainer_args_in)

trainer = Seq2SeqTrainer(model=model, args=training_args[0], train_data=train_data, tokenizer=tokenizer, data_collator=data_collator)

train_result = trainer.train(resume_from_checkpoint=False)
trainer.save_model()