"""
Common choices:

One-hot encoding (20 amino acids → 20-dim vector per residue).

Embedding layer (learn dense representation).

Pretrained protein embeddings (like ESM-2, ProtBERT, etc.).
"""
import os
import torch
import esm
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import logging
import pandas as pd

from sklearn.metrics import roc_auc_score
from training import MLPClassifier, load_and_prepare_dataset
from utils import load_mlp_classifier

mlp_model = None
embeddings_dict = None

MLP_MODEL_PATH = 'MCL1_10_mlp_classifier.pth'


class CrossAttentionClassifier(nn.Module):
    def __init__(self, emb_dim, num_heads=4, hidden_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # assume pep1, pep2: (batch, 1, emb_dim)
        pep1, pep2 = torch.split(x, self.emb_dim, dim=1)
        pep1, pep2 = (pep1, pep2)
        out, _ = self.cross_attn(pep1, pep2, pep2)  # pep1 attends to pep2
        return self.fc(out.squeeze(1))


class SiameseMLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        # Shared encoder for both peptides
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        pep1, pep2 = torch.split(x, self.emb_dim, dim=1)
        z1 = self.encoder(pep1)
        z2 = self.encoder(pep2)
        # combine: concat, diff, and abs-diff (common trick)
        combined = torch.cat([z1, z2], dim=1)
        return self.classifier(combined)

def covert_target_protein_seq_to_embedding(target_seq):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    input_data = []
    embeddings_dict = {}
    input_data.append(('target0', target_seq))

    batch_labels, batch_strs, batch_tokens = batch_converter(input_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    for seq_info, seq_repr in zip(input_data, sequence_representations):
        seq_name = seq_info[1]
        if seq_name not in embeddings_dict:
            embeddings_dict[seq_name] = seq_repr.numpy()
    return embeddings_dict

def covert_cyclic_peptide_seq_to_embedding(data_df, embedding_name):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    embeddings_dict = {}
    total_length = len(data_df)
    for i in tqdm(range(total_length)):
        row = data_df.iloc[i]
        peptides_names = row['seqA'], row['seqB']
        input_data = []
        for idx, seq in enumerate(peptides_names):
            if seq not in embeddings_dict:
                input_data.append((f'peptide{idx}', seq))
        if input_data:
            batch_labels, batch_strs, batch_tokens = batch_converter(input_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
            for seq_info, seq_repr in zip(input_data, sequence_representations):
                seq_name = seq_info[1]
                if seq_name not in embeddings_dict:
                    embeddings_dict[seq_name] = seq_repr.numpy()
            with open(os.path.join(embedding_file_prefix, embedding_name), 'wb') as f:
                pickle.dump(embeddings_dict, f)
        else:
            continue
    return embeddings_dict

    # # Look at the unsupervised self-attention map contact predictions
    # import matplotlib.pyplot as plt
    # for (_, seq), tokens_len, attention_contacts in zip(input_data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    #     plt.title(seq)
    #     plt.show()


def mlp_classifer_train_loop(cyclic_df,
                             embeddings_dict,
                             input_dim,
                             data_type,
                             cycle,
                             target_seq):
    total_length = len(cyclic_df)
    BATCH_SIZE = 64
    # ==============================
    # Setup Logging
    # ==============================
    import time
    start_time = time.time()
    log_file = f"training_log_{start_time}.log"
    # remove old log if exists
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    data_inputs, data_y = [], []

    invalid_count = 0
    for i in tqdm(range(total_length)):
        row = cyclic_df.iloc[i]
        pep_sq1, pep_sq2 = row['seqA'], row['seqB']
        sq1_plddt, sq2_plddt = row["seqA_plddt"], row["seqB_plddt"]
        if sq1_plddt > sq2_plddt:
            y = 1
        else:
            y = 0
        try:
            pep_pert_emb = embeddings_dict.get(pep_sq1, None)
            pep_mon_emb = embeddings_dict.get(pep_sq2, None)

            if pep_pert_emb is None or pep_mon_emb is None:
                pep_pert_tensor = torch.randn(1, len(pep_pert_emb))
                pep_mon_tensor = torch.randn(1, len(pep_mon_emb))

            # Convert to tensors and run inference
            pep_pert_tensor = torch.tensor(pep_pert_emb, dtype=torch.float32).unsqueeze(0)
            pep_mon_tensor = torch.tensor(pep_mon_emb, dtype=torch.float32).unsqueeze(0)

            # prepare target protein
            target_protein_embedding_dict = covert_target_protein_seq_to_embedding(target_seq)
            target_tensor = torch.tensor(target_protein_embedding_dict['target0'], dtype=torch.float32).unsqueeze(0)

            # Concatenate embeddings and run model
            inputs = torch.add(torch.sub(pep_pert_tensor, pep_mon_tensor), target_tensor)

            # inputs = torch.cat([pep_pert_tensor, pep_mon_tensor], dim=1)
            data_inputs.append(inputs)
            data_y.append(y)

        except:
            invalid_count += 1
            print(f"Error during embedding extraction, invalid sequence {pep_sq1}, invalid sequence {pep_sq2}")
            continue

    # ==============================
    # Dummy Training Data
    # ==============================
    X = torch.cat(data_inputs, dim=0).float()

    y = torch.tensor(data_y, dtype=torch.float32)
    VAL_SPLIT = 0.2

    dataset = torch.utils.data.TensorDataset(X, y)
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ==============================
    # Training Setup
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ==============================
    # Training Loop
    # ==============================
    epochs = 15
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        correct, total = 0, 0
        all_probs, all_labels = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Predictions & Metrics
            probs = torch.sigmoid(outputs).detach().cpu()
            preds = (probs > 0.5).float()
            correct += (preds == batch_y.cpu()).sum().item()
            total += batch_y.size(0)

            all_probs.extend(preds.numpy())
            all_labels.extend(batch_y.cpu().numpy())

        acc = correct / total
        avg_loss = epoch_loss / len(train_loader)
        roc_auc = roc_auc_score(all_labels, all_probs)

        model.eval()
        val_loss = 0.0
        all_val_probs, all_val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.unsqueeze(1).to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                all_val_probs.extend((outputs > 0.5).float().detach().cpu())
                all_val_labels.extend(batch_y.detach().cpu())
        val_loss /= len(val_loader)
        val_roc_auc = roc_auc_score(all_val_labels, all_val_probs)

        log_msg = (f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {roc_auc:.4f} |"
                   f" VAL AUC: {val_roc_auc:.4f}")
        print(log_msg)
        logger.info(log_msg)

    # ==============================
    # Save Model
    # ==============================
    torch.save(model.state_dict(), f"{data_type}_10_mlp_classifier_{cycle}.pth")
    logger.info("Training finished. Model saved to mlp_classifier.pth")


def mlp_classifer_infer_loop(cyclic_df, embeddings_dict, input_dim):
    mlp_model = MLPClassifier(input_dim)
    mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=torch.device("cpu")))
    mlp_model.eval()
    all_probs = []
    total_length = len(cyclic_df)
    for i in tqdm(range(total_length)):
        row = cyclic_df.iloc[i]
        keywords = row["keywords"]
        peptides_names = keywords.split("|")
        pep_sq1, pep_sq2 = peptides_names[0], peptides_names[1]
        # if pep_sq1 == "SCDHFEPGLVCE" and pep_sq2 == "SELNTQPGRVFR":
        #     print(row["seqA_plddt"], row["seqB_plddt"])
        sq1_plddt, sq2_plddt = row["seqA_plddt"], row["seqB_plddt"]
        pep_emb1 = embeddings_dict.get(pep_sq1, None)
        pep_emb2 = embeddings_dict.get(pep_sq2, None)

        if pep_emb1 is None or pep_emb2 is None:
            print(f"Warning: Missing embeddings for genes: {pep_emb1}, {pep_emb2}")
            return 0.5

        # Convert to tensors and run inference
        pep_sq1_tensor = torch.tensor(pep_emb1, dtype=torch.float32).unsqueeze(0)
        pep_sq2_tensor = torch.tensor(pep_emb2, dtype=torch.float32).unsqueeze(0)

        # minus embeddings and run model
        inputs = torch.sub(pep_sq1_tensor, pep_sq2_tensor)

        with torch.no_grad():
            logits = mlp_model(inputs)
            probability = torch.sigmoid(logits).item()
            all_probs.append(probability)

    return all_probs



def parser_cycle_data(cycle, data_type):
    from itertools import combinations
    random_select_counts = 5
    all_df = None
    for i in range(cycle):
        data_path_prefix = f'./data/cycle_{i+1}'
        data_name = f'{data_type}_design_sequences_plddt_{i}.csv'
        df = pd.read_csv(os.path.join(data_path_prefix, data_name))
        all_df = pd.concat([all_df, df], ignore_index=True)
    all_df = all_df.drop_duplicates()
    # Create all pairwise combinations
    pairs = []
    import random
    all_sequences = list(all_df['generate_seq'])
    total_length = len(all_sequences)
    for i in tqdm(range(total_length)):
        seqA = all_df.iloc[i]['generate_seq']
        for _ in range(random_select_counts):
            seqB = random.choice(all_sequences)
            seqA_plddt = all_df.iloc[i]['plddt']
            seqB_plddt = all_df[all_df['generate_seq'] == seqB]['plddt'].item()
            pairs.append({
                "seqA": seqA,
                "seqA_plddt": seqA_plddt,
                "seqB": seqB,
                "seqB_plddt": seqB_plddt,
            })

    # Convert to dataframe
    pairwise_df = pd.DataFrame(pairs)

    # Save to CSV
    pairwise_df.to_csv(
        os.path.join(data_path_prefix, f'{data_type}_design_sequence_pairwise_{cycle - 1}.csv'), index=False)

    return pairwise_df


if __name__ == '__main__':
    data_type = 'MDM2'
    cycle = 2
    protein_sequences = ''

    cyclic_path = ['/data3/zhangyu/own_data/rchem/MDM2_10_train_100k.csv']
    embedding_file_prefix = f'./data/cycle_{cycle}/'
    data_df = load_and_prepare_dataset(cyclic_path)
    base_df = data_df[['seqA', 'seqB', 'seqA_plddt', 'seqB_plddt']]
    # test_data_df = load_and_prepare_dataset(cyclic_path)
    embedding_name = f"{data_type}_10_train_esm_embedding_cycle_{cycle}.pkl"
    if cycle >= 1:
        cycle_df = parser_cycle_data(cycle, data_type)
        merged_df = pd.concat([cycle_df, base_df], ignore_index=True)
        if not os.path.exists(os.path.join(embedding_file_prefix, embedding_name)):
            embeddings_dict = covert_cyclic_peptide_seq_to_embedding(merged_df, embedding_name)
        else:
            with open(os.path.join(embedding_file_prefix, embedding_name), "rb") as f:
                embeddings_dict = pickle.load(f)


        input_dim = len(next(iter(embeddings_dict.values())))
        mlp_classifer_train_loop(merged_df, embeddings_dict, input_dim, data_type, cycle, protein_sequences)
        mlp_classifer_infer_loop(merged_df, embeddings_dict, input_dim, protein_sequences)
