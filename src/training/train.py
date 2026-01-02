import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configurações
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_FILE = "dataset/train.jsonl"
OUTPUT_DIR = "models/checkpoints"

def main():
    print("--- Iniciando Configuração do Treinamento ---")
    
    # 1. Configuração 4-bit (Quantização)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    print(f"Carregando modelo base: {MODEL_NAME}...")
    
    # 2. Carregar Modelo
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 3. Preparar modelo para treinamento 4-bit
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # 4. Carregar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 5. Configuração LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0,
        r=16, 
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # 6. Aplicar LoRA ao modelo
    model = get_peft_model(model, peft_config)
    print("Parâmetros treináveis:")
    model.print_trainable_parameters()
    
    # 7. Carregar Dataset
    try:
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
        print(f"Dataset carregado: {len(dataset)} exemplos.")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar {DATA_FILE}.")
        print(f"Detalhes: {e}")
        return
    
    # 8. Tokenizar dataset
    def tokenize_function(examples):
        # Combinar prompt e completion
        texts = [f"{p} {c}{tokenizer.eos_token}" for p, c in zip(examples['prompt'], examples['completion'])]
        
        # Tokenizar com padding para max_length
        result = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",  # Padding fixo para evitar problemas
        )
        
        # Labels são os mesmos que input_ids
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    print("Tokenizando dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # 9. Data Collator (sem padding adicional, já fizemos acima)
    from transformers import default_data_collator
    data_collator = default_data_collator
    
    # 10. Configuração do Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1, 
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True, 
        bf16=False,
        max_grad_norm=0.3,
        max_steps=50, 
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        save_total_limit=2,
    )
    
    # 11. Trainer padrão (mais estável que SFTTrainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 12. Executar Treinamento
    print("🚀 Iniciando Treinamento...")
    trainer.train()
    
    # 13. Salvar
    print(f"💾 Salvando modelo em {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Treinamento Concluído!")

if __name__ == "__main__":
    main()
