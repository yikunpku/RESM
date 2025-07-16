from typing import List, Any, Optional, Collection, Tuple, Dict
from pathlib import Path
import torch
from evo.ffindex import MSAFFindex
from evo.tokenization import Vocab
from evo.typed import PathLike
from evo.dataset import CollatableVocabDataset, NPZDataset, JsonDataset, A3MDataset, PickleDataset
from evo.tensor import collate_tensors
import rna_esm
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
from Bio import SeqIO

class IDTokenDataset(CollatableVocabDataset):
    def __init__(self, vocab, data_path):
        super().__init__(vocab)
        self.data = list(SeqIO.parse(data_path, "fasta"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = self.data[index]
        rna_id = item.id
        seq = item.seq
        one_hot_feat = self.one_hot_encode(seq)
        pairwise_onehot = torch.from_numpy(self.outer_concatenation(one_hot_feat, one_hot_feat))
        token = torch.from_numpy(self.vocab.encode_single_sequence(seq))
        return rna_id, str(seq), pairwise_onehot, token

    def one_hot_encode(self, sequences):
        sequences_arry = np.array(list(sequences)).reshape(-1, 1)
        lable = np.array(list('ACGU')).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(lable)
        seq_encode = enc.transform(sequences_arry).toarray()
        return seq_encode

    def outer_concatenation(self, matrix_A, matrix_B):
        "l1xn--> l1xl1x2n"
        matrix_A = matrix_A
        matrix_B = matrix_B
        matrix_C = np.zeros((matrix_A.shape[0], matrix_B.shape[0], matrix_A.shape[1] + matrix_B.shape[1]))
        for v1_dx, v1 in enumerate(matrix_A):
            for v2_dx, v2 in enumerate(matrix_B):
                v12 = np.hstack([v1, v2])
                matrix_C[v1_dx, v2_dx] = v12
        return matrix_C

class SSDataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            label_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)
        self.rnaids = split_files
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method  # "fast", "best"
        )
        self.npz_data = NPZDataset(
            data_path / label_path, split_files=split_files, lazy=True
        )
        assert len(self.a3m_data) == len(self.npz_data)

    def get(self, key: str):

        msa = self.a3m_data.get(key)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        missing_nt_index = torch.from_numpy(self.npz_data[key]['missing_nt_index'])
        contacts = torch.from_numpy(self.npz_data[key]['olabel'])
        return tokens, contacts, missing_nt_index

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rnaid = self.rnaids[index]
        msa = self.a3m_data[index]
        one_hot_feat = self.one_hot_encode(msa)
        pairwise_onehot = torch.from_numpy(self.outer_concatenation(one_hot_feat, one_hot_feat))
        tokens = torch.from_numpy(self.vocab.encode(msa))
        contacts = torch.from_numpy(self.npz_data[index]['olabel']).float()
        missing_nt_index = torch.from_numpy(self.npz_data[index]['missing_nt_index']).type(torch.long)
        return rnaid, pairwise_onehot, tokens, contacts, missing_nt_index

    def one_hot_encode(self, sequences):
        sequences_arry = np.array(list(sequences)).reshape(-1, 1)
        lable = np.array(list('ACGU')).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(lable)
        seq_encode = enc.transform(sequences_arry).toarray()
        return seq_encode

    def outer_concatenation(self, matrix_A, matrix_B):
        "l1xn--> l1xl1x2n"
        matrix_A = matrix_A
        matrix_B = matrix_B
        matrix_C = np.zeros((matrix_A.shape[0], matrix_B.shape[0], matrix_A.shape[1] + matrix_B.shape[1]))
        for v1_dx, v1 in enumerate(matrix_A):
            for v2_dx, v2 in enumerate(matrix_B):
                v12 = np.hstack([v1, v2])
                matrix_C[v1_dx, v2_dx] = v12
        return matrix_C

    def collater(
            self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        rnaid, tokens, contacts, missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'rna_id': rnaid,
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result


class RNADataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)

        self.rna_id = split_files
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method,
        )

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rna_id = self.rna_id[index]
        msa = self.a3m_data[index]
        tokens = torch.from_numpy(self.vocab.encode(msa))

        return rna_id, tokens


class LMDataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            label_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)
        self.rnaids = split_files
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method  # "fast", "best"
        )
        self.npz_data = NPZDataset(
            data_path / label_path, split_files=split_files, lazy=True
        )
        assert len(self.a3m_data) == len(self.npz_data)

    def get(self, key: str):
        msa = self.a3m_data.get(key)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        missing_nt_index = torch.from_numpy(self.npz_data[key]['missing_nt_index'])
        contacts = torch.from_numpy(self.npz_data[key]['olabel'])
        return tokens, contacts, missing_nt_index

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rnaid = self.rnaids[index]
        msa = self.a3m_data[index]
        # msa = FastaDataset.rna_trans_protein(msa)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        contacts = torch.from_numpy(self.npz_data[index]['olabel']).float()
        missing_nt_index = torch.from_numpy(self.npz_data[index]['missing_nt_index']).type(torch.long)
        return rnaid, tokens, contacts, missing_nt_index

    def collater(
            self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        rnaid, tokens, contacts, missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'rna_id': rnaid,
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result


class TSDataset(CollatableVocabDataset):
    def __init__(
        self,
        data_path: PathLike,
        msa_path: PathLike,
        vocab: Vocab,
        split_files: Optional[Collection[str]] = None,
        json_file: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = 1,
        sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method  #"hhfilter", "sample-weights", "diversity-max", "diversity-min"
        )
        self.json_data = JsonDataset(
            data_path=data_path,
            split_files=split_files,
            json_file=json_file,
        )
        self.rna_id = split_files
        assert len(self.a3m_data) == len(self.json_data)


    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rna_id = self.rna_id[index]
        msa = self.a3m_data[index]
        one_hot_feat = self.one_hot_encode(msa)
        pairwise_onehot = torch.from_numpy(self.outer_concatenation(one_hot_feat, one_hot_feat))
        tokens = torch.from_numpy(self.vocab.encode(msa))
        contacts = torch.tensor(self.json_data[index][3]).float()
        missing_nt_index = torch.tensor(self.json_data[index][4]).type(torch.long)
        return rna_id, pairwise_onehot, tokens, contacts, missing_nt_index

    def one_hot_encode(self, sequences):
        sequences_arry = np.array(list(sequences)).reshape(-1, 1)
        lable = np.array(list('ACGU')).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(lable)
        seq_encode = enc.transform(sequences_arry).toarray()
        return seq_encode

    def outer_concatenation(self, matrix_A, matrix_B):
        "l1xn--> l1xl1x2n"
        matrix_A = matrix_A
        matrix_B = matrix_B
        matrix_C = np.zeros((matrix_A.shape[0], matrix_B.shape[0], matrix_A.shape[1] + matrix_B.shape[1]))
        for v1_dx, v1 in enumerate(matrix_A):
            for v2_dx, v2 in enumerate(matrix_B):
                v12 = np.hstack([v1, v2])
                matrix_C[v1_dx, v2_dx] = v12
        return matrix_C

    def collater(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        tokens,contacts,missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result


class HardTSDataset(CollatableVocabDataset):
    def __init__(
        self,
        data_path: PathLike,
        vocab: Vocab,
        split_files: Optional[Collection[str]] = None,
        json_file: Optional[Collection[str]] = None,

    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        self.json_data = JsonDataset(
            data_path=data_path,
            split_files=split_files,
            json_file=json_file,
        )
        self.rna_id = split_files

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, index):
        rna_id = self.rna_id[index]
        rna_seq = self.json_data[index][1]
        one_hot_feat = self.one_hot_encode(rna_seq)
        pairwise_onehot = torch.from_numpy(self.outer_concatenation(one_hot_feat, one_hot_feat))
        tokens = torch.from_numpy(self.vocab.encode(rna_seq))
        contacts = torch.tensor(self.json_data[index][3]).float()
        missing_nt_index = torch.tensor(self.json_data[index][4]).type(torch.long)
        return rna_id, pairwise_onehot, tokens, contacts, missing_nt_index

    def one_hot_encode(self, sequences):
        sequences_arry = np.array(list(sequences)).reshape(-1, 1)
        lable = np.array(list('ACGU')).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(lable)
        seq_encode = enc.transform(sequences_arry).toarray()
        return seq_encode

    def outer_concatenation(self, matrix_A, matrix_B):
        "l1xn--> l1xl1x2n"
        matrix_A = matrix_A
        matrix_B = matrix_B
        matrix_C = np.zeros((matrix_A.shape[0], matrix_B.shape[0], matrix_A.shape[1] + matrix_B.shape[1]))
        for v1_dx, v1 in enumerate(matrix_A):
            for v2_dx, v2 in enumerate(matrix_B):
                v12 = np.hstack([v1, v2])
                matrix_C[v1_dx, v2_dx] = v12
        return matrix_C

class BPRNADataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            label_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        label_path = Path(label_path)
        self.rnaids = split_files
        self.bprna = PickleDataset(
            data_file=data_path / label_path,
            split_files=split_files,
        )

    def __len__(self) -> int:
        return len(self.bprna)

    def __getitem__(self, index):
        rnaid = self.rnaids[index]
        seq = self.bprna[index]["seq"]
        seq = seq.upper()
        seq = re.sub(r"[T]", "U", seq)
        seq = re.sub(r"[RYKMSWBDHVN~]|\.|\*", "X", seq)
        tokens = torch.from_numpy(self.vocab.encode(seq))
        contacts = torch.from_numpy(self.bprna[index]['contact_map']).float()
        missing_nt_index = torch.empty(0, dtype=torch.long)
        return rnaid, tokens, contacts, missing_nt_index

    def collater(
            self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        rnaid, tokens, contacts, missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'rna_id': rnaid,
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result


class MSADataset(CollatableVocabDataset):
    def __init__(self, ffindex_path: PathLike):
        vocab = Vocab.from_esm_alphabet(
            rna_esm.data.Alphabet.from_architecture("MSA Transformer")
        )
        super().__init__(vocab)

        ffindex_path = Path(ffindex_path)
        index_file = ffindex_path.with_suffix(".ffindex")
        data_file = ffindex_path.with_suffix(".ffdata")
        self.ffindex = MSAFFindex(index_file, data_file)

    def __len__(self):
        return len(self.ffindex)

    def __getitem__(self, idx):
        msa = self.ffindex[idx]
        return torch.from_numpy(self.vocab.encode(msa))

    def collater(self, batch: List[Any]) -> Any:
        return collate_tensors(batch)


