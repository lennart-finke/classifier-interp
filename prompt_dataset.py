from datasets import load_dataset
import pandas as pd
import random
from typing import List, Dict, Any
import os


def extract_text(item: Dict) -> str:
    """Extract text from a dataset row"""
    if isinstance(item, dict):
        # Special handling for wildjailbreak dataset
        if "adversarial" in item:
            text = item["adversarial"]
            if text is None or text == "":
                text = item.get("vanilla")
        else:
            text = item.get(
                "text",
                item.get(
                    "prompt",
                    item.get("jailbreak_query", None),
                ),
            )
            if text is None:
                text = next(iter(item.values()))
    else:
        text = str(item)

    if not isinstance(text, str):
        text = str(text)

    return text.strip()


def load_dataset_texts(config: Dict[str, Any]) -> List[str]:
    """Load texts from a single dataset"""
    try:
        # Load dataset
        dataset_kwargs = {
            "split": config["split"],
            "streaming": False,  # Always load full dataset for better performance
        }

        # Add optional parameters if they exist
        if "delimiter" in config:
            dataset_kwargs["delimiter"] = config["delimiter"]
        if "keep_default_na" in config:
            dataset_kwargs["keep_default_na"] = config["keep_default_na"]

        if "config" in config:
            dataset = load_dataset(config["name"], config["config"], **dataset_kwargs)
        else:
            dataset = load_dataset(config["name"], **dataset_kwargs)

        # Verify column exists
        text_column = config.get("text_column", "text")
        if text_column not in dataset.column_names:
            raise ValueError(
                f"Column '{text_column}' not found in dataset {config['name']}. Available columns: {dataset.column_names}"
            )

        # Apply subsampling if specified
        if "subsample_ratio" in config and config["subsample_ratio"] < 1.0:
            num_samples = int(len(dataset) * config["subsample_ratio"])
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)

        # Extract texts
        texts = []
        for item in dataset:
            text = extract_text(item)
            if text:  # Only add non-empty texts
                texts.append(text)

        return texts

    except Exception as e:
        print(f"Error loading dataset {config['name']}: {str(e)}")
        raise


def create_prompt_dataset(
    dataset_configs: List[Dict[str, Any]], output_path: str = "data/prompts.csv"
) -> None:
    """Create and save the combined dataset"""
    all_texts = []

    # Load and combine all datasets
    for config in dataset_configs:
        texts = load_dataset_texts(config)
        all_texts.extend(texts)
        print(f"Loaded {len(texts)} texts from {config['name']}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame({"prompts": all_texts})

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(all_texts)} prompts to {output_path}")


if __name__ == "__main__":
    # Sample dataset configurations
    dataset_configs = [
        {
            "name": "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "config": "jailbreak_2023_12_25",
            "split": "train",
            "text_column": "prompt",
            "subsample_ratio": 1.0,
        }
    ] * 5 + [
        {
            "name": "Open-Orca/OpenOrca",
            "split": "train",
            "text_column": "question",
            "subsample_ratio": 0.05,
        },
        {
            "name": "allenai/wildjailbreak",
            "config": "train",
            "split": "train",
            "text_column": "adversarial",
            "subsample_ratio": 1.0,
            "delimiter": "\t",
            "keep_default_na": False,
        },
        {
            "name": "EddyLuo/JailBreakV_28K",
            "config": "JailBreakV_28K",
            "split": "JailBreakV_28K",
            "text_column": "jailbreak_query",
            "subsample_ratio": 1.0,
        },
    ]

    create_prompt_dataset(dataset_configs)
