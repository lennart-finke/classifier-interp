from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE
import pandas as pd
import os
import multiprocessing
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Layers to train on (e.g., all layers 0-12 for a 13-layer model)
LAYERS = list(range(13))

# Tuned for an AMD Radeon Pro W7900 with 48GB of VRAM
BATCH_SIZE = 128 if device.type == "cuda" else 8

STEPS = 100001

model_name = "meta-llama/Prompt-Guard-86M"

# Load model and tokenizer using Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for model weights
)
model.to(device)
model.eval()

# Enable gradient checkpointing if using CUDA
if device.type == "cuda":
    model.gradient_checkpointing_enable()


class CSVDataIterator:
    def __init__(self, csv_path: str):
        """Initialize iterator for CSV file containing prompts"""
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.current_idx = 0
        self.total_rows = len(self.df)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.total_rows:
            raise StopIteration

        text = self.df.iloc[self.current_idx]["prompts"]
        self.current_idx += 1
        return text


# Define a custom data iterator that yields hidden states
class HiddenStateIterator:
    def __init__(self, data, model, tokenizer, device, layer, batch_size=8):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer = layer
        self.batch_size = batch_size
        self.current_batch = None
        self.current_idx = 0
        self.start_time = time.time()
        self.total_tokens = 0
        self.last_print_time = time.time()
        self.print_interval = 5

        # Prefetch next batch
        self.next_batch = None
        self.next_batch_texts = None
        self._prefetch_next_batch()

    def _prefetch_next_batch(self):
        """Prefetch the next batch of texts."""
        batch_texts = []
        for _ in range(self.batch_size):
            try:
                text = next(self.data)
                if text.strip():
                    batch_texts.append(text)
            except StopIteration:
                if not batch_texts:
                    self.next_batch = None
                    self.next_batch_texts = None
                    return
                break

        if batch_texts:
            self.next_batch_texts = batch_texts
            # Tokenize in advance
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Process batch in advance
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            self.next_batch = outputs.hidden_states[self.layer]
        else:
            self.next_batch = None
            self.next_batch_texts = None

    def __iter__(self):
        return self

    def __next__(self):
        # If we've exhausted the current batch, get a new one
        if self.current_batch is None or self.current_idx >= len(self.current_batch):
            if self.next_batch is None:
                raise StopIteration

            # Use prefetched batch
            self.current_batch = self.next_batch
            self.current_idx = 0

            # Update token count
            if self.next_batch_texts is not None:
                self.total_tokens += sum(
                    len(text.split()) for text in self.next_batch_texts
                )

            # Print throughput stats periodically
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                elapsed = current_time - self.start_time
                tokens_per_second = self.total_tokens / elapsed
                print(f"\nThroughput: {tokens_per_second:.2f} tokens/second")
                self.last_print_time = current_time

            # Start prefetching next batch
            self._prefetch_next_batch()

        # Return one activation at a time
        activation = self.current_batch[self.current_idx]
        self.current_idx += 1
        return activation


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    for layer in LAYERS:
        print(f"\n===== Training SAE for layer {layer} =====\n")
        data = CSVDataIterator("data/prompts.csv")
        hidden_state_iterator = HiddenStateIterator(
            data, model, tokenizer, device, layer=layer, batch_size=BATCH_SIZE
        )
        trainer_cfg = {
            "trainer": StandardTrainer,
            "dict_class": AutoEncoder,
            "activation_dim": 768,  # DeBERTa's hidden size
            "dict_size": 4 * 768,
            "lr": 1e-3,
            "device": str(device),
            "steps": STEPS,
            "layer": layer,
            "lm_name": model_name,
            "warmup_steps": 5000,
            "decay_start": 10000,
            "sparsity_warmup_steps": 5000,
        }

        # Get wandb configuration from environment variables
        wandb_entity = os.getenv("WANDB_ENTITY")
        wandb_project = os.getenv("WANDB_PROJECT")

        if not wandb_entity or not wandb_project:
            raise ValueError("WANDB_ENTITY and WANDB_PROJECT must be set in .env file")

        ae = trainSAE(
            data=hidden_state_iterator,
            trainer_configs=[trainer_cfg],
            steps=STEPS,
            use_wandb=True,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            log_steps=100,
            save_steps=[50000, 100000],
            save_dir=f"sae-test/layer_{layer}",
            activations_split_by_head=True,
            run_cfg={
                "model_name": model_name,
                "batch_size": BATCH_SIZE,
                "gradient_checkpointing": device.type == "cuda",
            },
            autocast_dtype=torch.bfloat16,
            device=device.type if device.type != "mps" else "cpu",
            normalize_activations=True,
        )
