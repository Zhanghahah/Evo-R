# # Training Rbio (A Demo)
# In this script we will demonstrate how to train Rbio on perturbation data obtained from
# the PertQA dataset available originally published here https://github.com/genentech/PerturbQA and adapted to our use case.
#
# Rbio implements LLM post-training using soft-verification mechanisms so that knowledge from biology models such as a virtual cell model (VCM) can be distilled and used within the LLM, rather than relying on hard ground truth labels obtained experimentally which are usually scarce and often costly.
#
# In this example, we use a simplified "VCM" consisting of a Multi Layer Perceptron (MLP) trained to answer perturbation questions. It exposes an interface that returns a probability when prompted with two gene names. This is the probability that a knockout of gene_a is having an effect on the expression of gene_b.
#
# We use this signal as a soft verification signal within our reward mechanism in order to post-train our LLM. This improves the LLM capabilities to answer questions of the form "Is a knockdown of <gene_a> in <cell_line> cells likely to result in differential expression of gene_b?"

# ## Imports, global variables, random seeds

import os
import logging
from typing import List

import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from collections import defaultdict
from trl import GRPOTrainer
import esm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType

from rewards import *

from utils import (
    set_random_seeds,
    load_mlp_classifier,
    setup_model_and_tokenizer,
    create_training_config,
    mlp_classifier_inference,
    peptide_mlp_binary_infer
)
import subprocess
import numpy as np

# Disabling logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

# Training configuration
# MODEL_NAME = "/data1/zhangyu/own_data/public_models/Qwen2.5-7B-Instruct"
MODEL_NAME = "./checkpoints/MDM2_cyclic_binding_design_2/checkpoint-5000"
N_STEPS = 5000
BATCH_SIZE = 8
NUM_GENERATIONS = 4
SAVE_EVERY = 1000
OUTPUT_DIR = "./checkpoints/MDM2_cyclic_binding_design_3/"

# Global step counter
STEP_COUNT = 0
cycle = 2

SEQUENCES = defaultdict(int)
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
# MLP classifier configuration (mlp was not trained on k562 cells)
MLP_MODEL_PATH = os.path.join(parent_dir, f'data/cycle_{cycle}/MDM2_10_mlp_classifier_{cycle}.pth')
EMBEDDING_FILE = os.path.join(parent_dir, f'data/cycle_{cycle}/MDM2_10_train_esm_embedding_cycle_{cycle}.pkl')

# Dataset paths
DATASET_PATHS = [

    "MDM2_10_cyclic_binding_redesign_10_v2.csv"
]

# Set seeds globally
set_random_seeds(42)


def load_model_and_tokenizer(model_name):
    """
    Loads a model and tokenizer from a trained checkpoint

    Args:
        model_name: name of base model to load
        model_checkpoint: name of rbio model_checkpoint to use
        device: device to use
    """
    model_path = f'/data7/rchem_model_backup/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


class MLPClassifier(nn.Module):
    """Simple MLP classifier for gene pair classification"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.model(x)
        return result


class ScoreModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # output: scalar score
        )

    def forward(self, x):
        return self.net(x)


# # Global variables for MLP model and embeddings
# mlp_model = None
# embeddings_dict = None


# # Dataset
#
# `load_and_prepare_dataset` has the purpose of loading the training datasets (k562 only in this example) as a pandas dataframe.
#
# `create_mlp_labeled_dataset_generator` yields samples from this dataset that have been soft-labeled by our simplified VCM: the MLP defined above.

# In[ ]:


def load_and_prepare_dataset(dataset_paths: List[str], balance_pos_neg: bool = True) -> pd.DataFrame:
    """Load CSV datasets and combine them into a single DataFrame"""
    if len(dataset_paths) == 1:
        dataset_df = pd.read_csv(dataset_paths[0])
    else:
        dataset_list = []
        for path in dataset_paths:
            dataset_list.append(pd.read_csv(path))
        dataset_df = pd.concat(dataset_list, ignore_index=True)

    print(f"Loaded dataset with {len(dataset_df)} rows")
    return dataset_df


def create_mlp_labeled_dataset_generator(dataset_df: pd.DataFrame, tokenizer, balance_pos_neg: bool = True):
    """Generate training examples with MLP-based labeling"""

    CoT_suffix = "\n You will provide the reasoning step-by-step, using detailed biological knowledge from the training."
    if balance_pos_neg:
        # Use 2x the dataset length to ensure enough samples for training
        dataset_length = len(dataset_df) * 2
    else:
        dataset_length = len(dataset_df)

    for i in range(dataset_length):
        # Sample from dataset (with replacement for longer training)
        sample_idx = i % len(dataset_df)
        row = dataset_df.iloc[sample_idx]

        # Prepare sample data for MLP classification
        sample_data = {
            "system_prompt": row["system_prompt"] + CoT_suffix,
            "user_prompt": row["user_prompt"],
            "keywords": row["keywords"],
        }

        # Get MLP prediction
        # mlp_probability = peptide_mlp_binary_infer(sample_data)

        # Determine label based on MLP probability
        # predicted_label = 1 if mlp_probability > 0.5 else 0

        # Prepare sample with MLP-generated label
        sample = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            # "label": predicted_label,
            "classes": "no|yes",
            # "class_confidences": f"{1.0 - mlp_probability:.3f}|{mlp_probabilinty:.3f}",
            "keywords": row["keywords"],
            "task": row["task"],
            "candidate_plddt": row["candidate_plddt"],
            "candidate": row["candidate"]
        }

        # Format messages for chat template
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        yield {
            "prompt": prompt,
            # "label": sample["label"],
            # "classes": sample["classes"],
            # "class_confidences": sample["class_confidences"],
            "keywords": sample["keywords"],
            "task": sample["task"],
            "system_prompt": sample["system_prompt"],
            "user_prompt": sample["user_prompt"],
            "candidate_plddt": row["candidate_plddt"],
            "candidate": row["candidate"]
        }


def create_peptide_loop_dataset_generator(dataset_df: pd.DataFrame, tokenizer, balance_pos_neg: bool = True):
    """Generate training examples with MLP-based labeling"""
    if balance_pos_neg:
        # Use 2x the dataset length to ensure enough samples for training
        dataset_length = len(dataset_df) * 2
    else:
        dataset_length = len(dataset_df)

    for i in range(dataset_length):
        # Sample from dataset (with replacement for longer training)
        sample_idx = i % len(dataset_df)
        row = dataset_df.iloc[sample_idx]

        # Prepare sample data for MLP classification
        sample_data = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "keywords": row["keywords"]
        }

        # Get confidence
        pdb_probability = row['prob_score']

        # Get predict label
        # predicted_label = row['label']
        # probability = float(row['class_confidences'].split('|')[0])

        # Determine label based on MLP probability
        predicted_label = row['label']
        if predicted_label == 1:
            class_confidences = f"{1.0 - pdb_probability:.3f}|{pdb_probability:.3f}"
        elif predicted_label == 0:
            class_confidences = f"{pdb_probability:.3f}|{1.0 - pdb_probability:.3f}"
        # Prepare sample with MLP-generated label
        sample = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "label": predicted_label,
            "classes": "no|yes",
            "class_confidences": class_confidences,
            "keywords": row["keywords"],
            "task": row["task"],
            "mlp_probability": pdb_probability
        }

        # Format messages for chat template
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        yield {
            "prompt": prompt,
            "label": sample["label"],
            "classes": sample["classes"],
            "class_confidences": sample["class_confidences"],
            "keywords": sample["keywords"],
            "task": sample["task"],
            "system_prompt": sample["system_prompt"],
            "user_prompt": sample["user_prompt"],
        }

def reward_by_reasoner(
        completion,
        reward_model,
        reward_tokenizer,
        keywords,
        num_return_sequences=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    answer_text, answer_valid = extract_designed_sequence(completion)

    keyword_list = [k for k in keywords.split("|") if k]
    ref_seq = keyword_list[0]

    if answer_text is None:
        return 0.0

    # preference reasoning inference
    system_prompt_orig = "A conversation between User and Biologist. \
The user asks which of two predicted cyclic peptides is more likely to bind the given Mcl-1 target sequence, \
based on internal structure confidence and sequence features. The Biologist reasons carefully and answers. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, \
i.e., <think> reasoning process here </think> <answer> answer here </answer>."

    question = f"Based on the Mcl-1 target sequence GSDELYRQSLEIISRYLREQATGAKDTKPMGRSGATSRKALETLRRVGDGVQRNHETAFQGMLRKLDIKNEDDVKSLSRVMIHVFSDGVTNWGRIVTLISFGAFVAKHLKTINQESCIEPLAESITDVLVRTKRDWLVKQRGWDGFVEFFHVEDLEGG, \
is {answer_text} expected to exhibit a higher probability of binding effectively compared to {ref_seq}? Both peptides are considered cyclic peptide binders derived for this target. The answer is either yes or no."

    CoT_suffix = "\n You will provide the reasoning step-by-step, using detailed biological knowledge from the training."
    system_prompt = system_prompt_orig + CoT_suffix

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    texts = reward_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = reward_tokenizer(texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generated_ids = reward_model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            num_return_sequences=num_return_sequences,
            top_k=None,
        )

    input_ids = model_inputs.input_ids[0]
    input_len = input_ids.shape[0]
    generated_ids = [output_ids[input_len:] for output_ids in generated_ids]
    generated_texts = [
        tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
    ]
    postive_samples = 0
    answers = []
    for response in generated_texts:
        answer = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
        if len(answer) > 0:
            answer = answer[0].strip().lower()
        answers.append(answer)
        try:
            if "yes" in answer:
                postive_samples += 1
        except:
            print(answer)
            continue

    response_design_data_path = './tmp_seq_design.csv'
    responses_df = pd.DataFrame(
        {

            "design_seq": answer_text * num_return_sequences,
            "ref_seq": ref_seq * num_return_sequences,
            "answers": answers,
        }
    )
    responses_df.to_csv(
        response_design_data_path, mode='a',
        header=not os.path.exists(response_design_data_path),
        index=False
    )
    #
    return postive_samples / num_return_sequences


def reward_answer_against_label(completion: str, classes: str, class_confidence: str) -> float:
    """Compute reward based on whether answer matches expected label"""
    answer = extract_binary_answer(completion)
    if answer is None:
        return 0.0

    answer = "yes" if answer else "no"
    possible_classes = classes.split("|")
    confidences = [float(c) for c in class_confidence.split("|")]

    for label, conf in zip(possible_classes, confidences):
        if answer == label.strip().lower():
            return conf

    return 0.0


def composite_formatting_reward(text: str, use_go: bool = False) -> float:
    """Compute composite formatting reward based on multiple checks"""
    at_least_one_think = has_at_least_one_think(text)
    has_tags = has_any_tag(text)

    checks = [
        at_least_one_think,
        low_untagged_ratio(text),
        is_not_too_long(text),
        has_one_answer(text),
        answer_after_thinks(text),
        thinks_have_text(text) * at_least_one_think,
        no_nested_tags(text) * has_tags,
        has_limited_thinks(text) * at_least_one_think,
        all_tags_properly_closed(text) * has_tags,
        ends_with_answer(text),
        starts_with_think(text),
    ]

    # Remove start_with_think dependency if using GO ontology
    if use_go:
        checks = checks[:-1]

    return sum(checks) / len(checks)


def keywords_mentioned_in_think(text: str, keywords: str) -> float:
    """Check how many keywords are mentioned in think sections"""
    keyword_list = [k for k in keywords.split("|") if k]

    if not keyword_list:
        return 1.0

    think_contents = extract_think(text)
    if not think_contents:
        return 0.0

    found_keywords = 0
    for keyword in keyword_list:
        if keyword in think_contents:
            found_keywords += 1

    if found_keywords == 1:
        return 1.0
    else:
        return found_keywords / len(keyword_list)


def calulate_reward_by_plddt(design_plddt, ref_plddt, alpha=0.04):
    """
    Returns reward in (0,1) approximately. alpha controls sensitivity:
      - smaller alpha -> gentler slope (smaller diffs give small change)
      - larger alpha -> steeper response
    Default alpha=0.04 is a reasonable starting point.
    """
    import math
    diff = float(design_plddt) - float(ref_plddt)
    if diff < 0:
        return float(0.5 * math.tanh(alpha * diff))
    else:
        # map diff to [-inf,inf] then tanh to (-1,1). Scale so equal->0.5.

        return diff / 100 + float(0.5 * math.tanh(alpha * diff))


def positional_repetition_penalty(gen, ref):
    # count how many repeated sliding-window 3–4 aa motifs exist
    window_sizes = [2, 3, 4]

    repeats = 0
    for w in window_sizes:
        seen = {}
        for i in range(len(gen) - w + 1):
            motif = gen[i:i + w]
            seen[motif] = seen.get(motif, 0) + 1

        # count motifs repeated more than 2 times
        repeats += sum((c - 1) for c in seen.values() if c > 2)

    # convert to penalty in [0,1]
    penalty = 1 - np.tanh(repeats / 10)
    return penalty



def reward_by_mlp_confidence(generate_seq, ref_seq, ref_plddt):
    if mlp_model is None or embeddings_dict is None:
        raise RuntimeError("MLP model not loaded. Call load_mlp_classifier() first.")
    # feature extraction by ESM mopdel;
    input_data = []
    ref_emb = embeddings_dict.get(ref_seq, None)
    if ref_emb is None:
        input_data.append(("peptide0", ref_seq))

    input_data.append(("peptide1", generate_seq))
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(input_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    esm_model.eval()
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    if ref_emb is None:
        ref_emb = sequence_representations[0].unsqueeze(0).clone().detach()
    else:
        ref_emb = torch.tensor(ref_emb, dtype=torch.float32).unsqueeze(0)

    gen_emb = sequence_representations[-1].unsqueeze(0).clone().detach()
    mlp_input = torch.sub(gen_emb.clone().detach(), ref_emb.clone().detach())

    # mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=torch.device("cpu")))
    mlp_model.eval()
    with torch.no_grad():
        logits = mlp_model(mlp_input)
        probability = torch.sigmoid(logits).item()
    return probability


def save_expand_mode_simple(generate_seq, generate_seq_prob, ref_seq, ref_seq_plddt):
    out_csv = './tmp/design_sequences.csv'
    df = pd.DataFrame({
        "generate_seq": [generate_seq],
        "generate_seq_prob": [generate_seq_prob],
        "ref_seq": [ref_seq],
        "ref_seq_plddt": [ref_seq_plddt],
    })
    df.to_csv(out_csv,
              mode='a',
              header=not os.path.exists(out_csv),
              index=False)


def internal_repetition_penalty(seq):
    from collections import Counter
    counts = Counter(seq)
    max_ratio = max(counts[a] for a in counts) / len(seq)
    return -max_ratio


def compute_simple_reward(
        completions: List[str],
        keywords: List[str],
        candidate: str,
        candidate_plddt: str,
        **kwargs) -> List[float]:
    """Compute rewards for model completions using format, mention, and answer rewards"""
    scores = []
    generate_seqs = []

    global STEP_COUNT, mlp_model, embeddings_dict, esm_model, alphabet, SEQUENCES

    for completion, keyword_list, reference, ref_plddt in zip(
            completions, keywords, candidate, candidate_plddt
    ):
        # Format reward: checks proper tag structure
        format_reward = composite_formatting_reward(completion, use_go=False)

        # Mention reward: checks if keywords are mentioned in think sections
        mention_reward = keywords_mentioned_in_think(completion, keyword_list)

        # Answer reward: checks if answer matches expected label
        # answer_reward = reward_answer_against_label(completion, class_list, confidences)
        generate_seq, answer_valid = extract_designed_sequence(completion)
        if generate_seq is not None:
            if len(generate_seq) > 15 or len(generate_seq) < 5: # limi
                design_reward, diversity_reward = 0.0, 0.0
            else:
                diversity_reward = positional_repetition_penalty(generate_seq, reference)
                # answer reward # 2: check if the gerated sequences show higher performance.
                design_reward = reward_by_mlp_confidence(generate_seq, reference, ref_plddt)
                # Answer reward non_symmetric
                # scaled_design_reward = reward_non_symmetric(design_reward)
        else:
            design_reward = 0.0
            diversity_reward = 0.0

        # repetition too many times should be punished
        freq = SEQUENCES[generate_seq] // NUM_GENERATIONS
        repetition_penality = float(freq) / (freq + NUM_GENERATIONS)

        # Combine rewards (answer reward gets 2x weight as it's most important)
        total_score = (0.5 * format_reward +
                       2.5 * design_reward +
                       0.25 * diversity_reward +
                       0.5 * mention_reward -
                       0.25 * repetition_penality)

        scores.append(total_score)

        # save sequences
        SEQUENCES[generate_seq] = design_reward


        # Debug prints every 100 steps to monitor model outputs
        if STEP_COUNT % 50 == 0:
            logger.info("\n" + "=" * 80)
            logger.info(f"DEBUG: Sample {STEP_COUNT} - Step {STEP_COUNT}")
            logger.info("=" * 80)

            # Print the completion to see what the model generated
            logger.info(f"MODEL OUTPUT:")
            logger.info(f"{completion}")
            logger.info("-" * 80)

            # Print reward breakdown
            logger.info(f"REWARD BREAKDOWN:")
            logger.info(f"  Format reward: {format_reward:.3f}")
            logger.info(f"  Mention reward: {mention_reward:.3f}")
            logger.info(f"  Answer reward: {design_reward:.3f}")
            logger.info(f"  Diversity reward: {diversity_reward:.3f}")
            logger.info(f"  Repetition penality: {repetition_penality:.3f}")
            logger.info(f"  Total score: {total_score:.3f}")
            logger.info("-" * 80)

            # Print expected vs predicted
            logger.info(f"EXPECTED vs PREDICTED:")
            logger.info(f"  Keywords: {keyword_list}")
            logger.info("-" * 80)

            # Print reward details
            logger.info(f"REWARD DETAILS:")
            logger.info(f"  Answer extraction: {extract_binary_answer(completion)}")
            logger.info(f"  Think content: {extract_think(completion)[:100]}...")
            logger.info("=" * 80 + "\n")

            df = pd.DataFrame({
                "generate_seq": list(SEQUENCES.keys()),
                "generate_seq_prob": list(SEQUENCES.values()),

            })
            df.to_csv(f'./tmp/MDM2_mlp_design_sequences_{cycle}.csv',
                      index=True)

    STEP_COUNT += 1

    return scores


# # Training

if __name__ == '__main__':
    import time

    start_time = time.time()
    log_file = f"./logging/training_log_{start_time}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    print("Starting RChem training with streaming prob labeling...")

    # Load and prepare dataset
    print("Loading dataset...")
    reward_model_name = "MCL1_train_sampled_200/checkpoint-15000"
    dataset_df = load_and_prepare_dataset(DATASET_PATHS)

    # Load MLP classifier
    print("Loading Score ranking classifier...")
    mlp_model, embeddings_dict = load_mlp_classifier(MLP_MODEL_PATH, EMBEDDING_FILE, MLPClassifier)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)  # Configure LoRA for mathematical reasoning adaptation
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    lora_config = LoraConfig(
        r=16,  # Rank: adaptation capacity (16 good for reasoning tasks)
        lora_alpha=32,  # Scaling factor (typically 2x rank)
        target_modules=["q_proj", "v_proj"],  # Focus on attention query/value for reasoning
        lora_dropout=0.1,  # Regularization to prevent overfitting
        bias="none",  # Skip bias adaptation for simplicity
        task_type=TaskType.CAUSAL_LM,  # Causal language modeling task
    )
    print(" Applying LoRA adaptation to model...")
    model = get_peft_model(model, lora_config)

    print("LoRA Training Parameters Summary:")
    model.print_trainable_parameters()  # Shows trainable vs total parameters

    # Create streaming dataset generator
    print("Creating streaming dataset generator...")
    dataset = Dataset.from_generator(
        create_mlp_labeled_dataset_generator,  # create_peptide_loop_dataset_generator
        gen_kwargs={
            "dataset_df": dataset_df,
            "tokenizer": tokenizer,
            "balance_pos_neg": False,

        },
        cache_dir="./peptide_tmp_generator",
    )

    # Create training configuration
    print("Setting up training configuration...")
    training_config = create_training_config(
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        max_steps=N_STEPS,
        save_every=SAVE_EVERY
    )

    # Create trainer
    print("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=compute_simple_reward,
        args=training_config,
        train_dataset=dataset,
    )

    # Start training
    print(f"Starting training for {N_STEPS} steps...")
    trainer.train()

    print("Training completed!")
