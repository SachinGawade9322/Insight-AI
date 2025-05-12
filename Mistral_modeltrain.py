import os
import json
import torch
import numpy as np
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.utils import logging
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.metrics import accuracy_score, f1_score
import evaluate

logging.set_verbosity_info()

MODEL_NAME = "bitext/Mistral-7B-Banking-v2"
OUTPUT_DIR = "./fintech_model_output"
TRAINING_DATA_PATH = r"C:\Users\Admin\Desktop\Banking_chatbot\caterlyAI\Fintech\output\output.json"
CACHE_DIR = "./cache"
OFFLOAD_FOLDER = "./offload_folder"
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
MAX_SEQ_LENGTH = 1023
LOG_STEPS = 5
SAVE_STEPS = 0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
SEED = 42

os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


torch.manual_seed(SEED)
np.random.seed(SEED)

def load_training_data(data_path):
    """
    Load and preprocess training data
    """
    print(f"Loading training data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    return data

def format_prompt(query):
    """
    Format the input prompt for the banking model
    """
    return f"""<s>[INST] You are a financial chatbot named Fintech. Your task is to analyze bank transactions based on user queries.
For each query, extract relevant transactions and categorize them with appropriate main and subcategories.
Also provide AI-powered insights about the spending patterns.

Query: {query} [/INST]"""

def format_training_data(examples):
    """
    Format training data for instruction fine-tuning
    """
    formatted_data = []
    
    for example in examples:
        query = example["query"]
        response = example["response"]
        
        formatted_example = {
            "text": f"{format_prompt(query)}\n\n{response}</s>"
        }
        formatted_data.append(formatted_example)
    
    return formatted_data

def prepare_dataset(formatted_data):
    """
    Create a HuggingFace dataset from formatted examples
    """

    dataset = Dataset.from_list(formatted_data)

    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['test'])}")
    
    return dataset

def preprocess_function(examples, tokenizer):
    """
    Tokenize examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics (placeholder - for demonstration purposes)
    """

    return {
        "accuracy": 0.85,  
        "f1": 0.82,        
    }

def load_model_and_tokenizer():
    """
    Load the pretrained model and tokenizer with GPU and quantization configuration.
    """
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB")
        else:
            print("No GPU available, using CPU")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "right"
          
        torch.cuda.empty_cache()
        
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            offload_folder=OFFLOAD_FOLDER
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )

        model = get_peft_model(model, lora_config)

        print_trainable_parameters(model)
        
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise

def print_trainable_parameters(model):
    """
    Print number of trainable parameters
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%})")
    print(f"Total parameters: {all_params:,}")

def train_model(model, tokenizer, dataset):
    """
    Train the model using LoRA fine-tuning
    """

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="no",
        eval_steps=SAVE_STEPS,
        save_strategy="no",
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=LOG_STEPS,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        fp16=True,
        optim="adamw_torch",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting model training...")
    trainer.train()

    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def create_model_card():
    """
    Create a model card with documentation
    """
    model_card = """
    # Fintech Chatbot Model

    This model is fine-tuned from bitext/Mistral-7B-Banking-v2 for financial transaction categorization and analysis.

    ## Model Description
    - **Base model:** bitext/Mistral-7B-Banking-v2
    - **Fine-tuning method:** LoRA (Low-Rank Adaptation)
    - **Training data:** Bank transaction data with categorization
    - **Primary task:** Transaction analysis and categorization

    ## Capabilities
    - Extract relevant transactions based on user queries
    - Categorize transactions into main categories and subcategories
    - Provide financial insights based on transaction patterns
    - Handle natural language queries about financial data

    ## Parameters
    - LORA_R: {LORA_R}
    - LORA_ALPHA: {LORA_ALPHA}
    - LEARNING_RATE: {LEARNING_RATE}
    - BATCH_SIZE: {BATCH_SIZE}
    - NUM_EPOCHS: {NUM_EPOCHS}
    """.format(
        LORA_R=LORA_R,
        LORA_ALPHA=LORA_ALPHA,
        LEARNING_RATE=LEARNING_RATE,
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPOCHS=NUM_EPOCHS
    )
    
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(model_card)

def test_model(tokenizer, model, test_query):
    """
    Test the model on a sample query
    """
    prompt = format_prompt(test_query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response.replace(prompt, "").strip()
    print("\nTest query:", test_query)
    print("\nModel response:", response)

# def main():
#     """
#     Main function to orchestrate the fine-tuning process
#     """
#     # Load training data
#     data = load_training_data(TRAINING_DATA_PATH)
    
#     # Format the data for training
#     formatted_data = format_training_data(data)
    
#     # Prepare dataset
#     dataset = prepare_dataset(formatted_data)
    
#     # Load model and tokenizer
#     model, tokenizer = load_model_and_tokenizer()
    
#     # Train the model
#     train_model(model, tokenizer, dataset)
    
#     # Create model card
#     create_model_card()
    
#     # Test the model with a sample query
#     test_query = "Can you analyze my Uber transactions for the last month?"
#     test_model(tokenizer, model, test_query)
    
#     print("Fine-tuning completed successfully!")

def main():
    """
    Main function to orchestrate the fine-tuning process
    """
    data = load_training_data(TRAINING_DATA_PATH)

    formatted_data = format_training_data(data)

    dataset = prepare_dataset(formatted_data)

    model, tokenizer = load_model_and_tokenizer()

    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]  
    )

    train_model(model, tokenizer, tokenized_dataset)

    create_model_card()

    test_query = "Can you analyze my Uber transactions for the last month?"
    test_model(tokenizer, model, test_query)
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()