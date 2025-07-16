import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import rna_esm
from evo.tokenization import Vocab, mapdict
from model import ESM2
from dataset import RNADataset
from dataclasses import dataclass
import argparse

# Set random seed for reproducibility
torch.manual_seed(42)

current_directory = Path(__file__).parent.absolute()


@dataclass
class DataConfig:
    pass


@dataclass
class OptimizerConfig:
    pass


@dataclass
class TrainConfig:
    pass


@dataclass
class TransformerConfig:
    pass


@dataclass
class LoggingConfig:
    pass


@dataclass
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: TransformerConfig = TransformerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    logging: LoggingConfig = LoggingConfig()
    fast_dev_run: bool = False
    resume_from_checkpoint: str = None
    val_check_interval: int = 1000


@dataclass
class InferenceConfig:
    architecture: str = "rna-esm"
    base_model: str = "RESM_150M"  # or "RESM_650M"
    data_path: str = str("./data")
    msa_path: str = str("./data/dsdata/msa")
    data_split: str = "extract_ss_data_alphaid.txt"
    model_path: str = None  # Will be set in __post_init__
    device: str = "cuda"
    output_dir: str = str( "./data/features")
    max_seqlen: int = 1024

    def __post_init__(self):
        # Set model parameters and path based on base model
        if self.base_model == "RESM_150M":
            self.embed_dim = 640
            self.num_attention_heads = 20
            self.num_layers = 30
            if self.model_path is None:
                self.model_path = "./ckpt/RESM-150M-KDNY.ckpt"
        elif self.base_model == "RESM_650M":
            self.embed_dim = 1280
            self.num_attention_heads = 20
            self.num_layers = 33
            if self.model_path is None:
                self.model_path = "./ckpt/RESM-650M-KDNY.pt"
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")


def extract_features(config: InferenceConfig) -> None:
    """Extract RNA embeddings and attention maps from RESM pre-trained model."""

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Initialize vocabularies and mapping
    if config.base_model == "RESM_150M":
        _, protein_alphabet = rna_esm.pretrained.esm2_t30_150M_UR50D()
    elif config.base_model == "RESM_650M":
        _, protein_alphabet = rna_esm.pretrained.esm2_t33_650M_UR50D()

    rna_alphabet = rna_esm.data.Alphabet.from_architecture(config.architecture)

    protein_vocab = Vocab.from_esm_alphabet(protein_alphabet)
    rna_vocab = Vocab.from_esm_alphabet(rna_alphabet)
    rna_to_protein = {'A': 'K', 'U': 'D', 'C': 'N', 'G': 'Y'}
    rna_map_dict = mapdict(protein_vocab, rna_vocab, rna_to_protein)
    rna_map_vocab = Vocab.from_esm_alphabet(rna_alphabet, rna_map_dict)

    # Load RNA IDs from split file
    with open(Path(config.data_path) / config.data_split) as f:
        rnas = f.read().splitlines()
        rnas.sort()

    # Initialize dataset
    dataset = RNADataset(
        config.data_path,
        config.msa_path,
        rna_map_vocab,
        split_files=rnas,
        max_seqs_per_msa=1,
    )

    # Initialize model
    @dataclass
    class TransformerConfig:
        embed_dim: int = config.embed_dim
        num_attention_heads: int = config.num_attention_heads
        num_layers: int = config.num_layers
        max_seqlen: int = config.max_seqlen
        dropout: float = 0.1
        attention_dropout: float = 0.1
        activation_dropout: float = 0.1
        attention_type: str = "standard"
        performer_attention_features: int = 256

    model = ESM2(
        vocab=protein_vocab,
        model_config=TransformerConfig(),
        optimizer_config=OptimizerConfig(),  # Use the defined class
        contact_train_data=None,
        token_dropout=False,  # Disable dropout for inference
    )

    # Load model weights
    model.load_state_dict(torch.load(
            config.model_path,
            map_location="cpu")['state_dict'], strict=True)
    model = model.eval()
    model = model.to(device)

    # Create output directories
    model_name = Path(config.model_path).stem
    save_emb_path = os.path.join(config.output_dir, model_name, "embedding")
    save_atp_path = os.path.join(config.output_dir, model_name, "attention_map")

    os.makedirs(save_emb_path, exist_ok=True)
    os.makedirs(save_atp_path, exist_ok=True)

    # Extract features
    with torch.no_grad():
        for rna_id, tokens in tqdm(dataset, desc="Extracting features"):
            tokens = tokens.unsqueeze(0).to(device)

            # Forward pass
            results = model(tokens, repr_layers=[config.num_layers], need_head_weights=True)

            # Extract attention maps
            attentions = results["attentions"]
            start_idx = int(rna_map_vocab.prepend_bos)
            end_idx = attentions.size(-1) - int(rna_map_vocab.append_eos)
            attentions = attentions[..., start_idx:end_idx, start_idx:end_idx]
            seqlen = attentions.size(-1)
            attentions = attentions.view(-1, seqlen, seqlen).cpu().numpy()

            # Save attention maps
            attentions_path = os.path.join(save_atp_path, f"{rna_id}_atp.npy")
            np.save(attentions_path, attentions)

            # Extract embeddings
            embedding = results["representations"][config.num_layers]
            start_idx = int(rna_map_vocab.prepend_bos)
            end_idx = embedding.size(-2) - int(rna_map_vocab.append_eos)
            embedding = embedding[:, start_idx:end_idx, :].squeeze(0).cpu().numpy()

            # Save embeddings
            embedding_path = os.path.join(save_emb_path, f"{rna_id}_emb.npy")
            np.save(embedding_path, embedding)

    print("Feature extraction completed!")


def main():
    parser = argparse.ArgumentParser(description="Extract features from RESM model")
    parser.add_argument("--base_model", type=str, default="RESM_150M",
                        choices=["RESM_150M", "RESM_650M"],
                        help="Base ESM2 model architecture")
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint (overrides default)")
    parser.add_argument("--output_dir", type=str, help="Output directory for features")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data_split", type=str, help="Split file name")

    args = parser.parse_args()

    # Create config with command line overrides
    config = InferenceConfig()
    if args.base_model:
        config.base_model = args.base_model
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.data_split:
        config.data_split = args.data_split
    # model_path should be set after base_model to allow override
    if args.model_path:
        config.model_path = args.model_path

    extract_features(config)


if __name__ == "__main__":
    main()