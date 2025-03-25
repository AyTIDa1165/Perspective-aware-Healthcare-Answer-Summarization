import pandas as pd
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import os

# Paths and device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
input_file = "/raid/home/mehul22294/french_dataset/instruction_set_test.tsv"
output_file = "/raid/home/mehul22294/french_dataset/instruction_set_gemma_zero_shot.tsv"

# Load the pre-trained base model   
max_seq_length = 2048
dtype = None  # Auto-detect dtype
# load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b",
    max_seq_length=max_seq_length,
    dtype=dtype,  
    load_in_4bit= False
)

print("Base Model Loaded")

FastLanguageModel.for_inference(model)

# Define the Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""


#Instruction in french
instruction_male_fr = "Sans changer la structure de la phrase, transformez cette phrase comme si elle était prononcée par un locuteur masculin."
instruction_female_fr = "Sans changer la structure de la phrase, transformez cette phrase comme si elle était prononcée par une locutrice féminine"

# Function to generate output for a given input text
def generate_output(wrong_ref_text,speaker_gender):
    instruction = ""
    if speaker_gender == "M":
        instruction = instruction_male_fr
    else:
        instruction = instruction_female_fr
    # Create input text using the Alpaca prompt

    input_text = alpaca_prompt.format(
        instruction,
        wrong_ref_text,
        ""
    )
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    # Generate output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )

    # Decode and return the result
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load the dataset
df = pd.read_csv(input_file, sep="\t")

# Create a new column for generated output
df["Generated_output"] = ""

for index, row in df.iterrows():
    if row["gender"].strip() == "MM" or row["gender"].strip() == "FM":
        wrong_ref = row["input"]
        generated_text = generate_output(wrong_ref, "M")
        df.at[index, "Generated_output"] = generated_text
        print(f"Processed ID {row['ID']}: {generated_text}")

    elif row["gender"].strip() == "FF" or row["gender"].strip() == "MF":
        wrong_ref = row["input"]
        generated_text = generate_output(wrong_ref,"F")
        df.at[index, "Generated_output"] = generated_text
        print(f"Processed ID {row['ID']}: {generated_text}")        

# Save the updated dataset to a new TSV file
df.to_csv(output_file, sep="\t", index=False)
print(f"Updated dataset saved to {output_file}")