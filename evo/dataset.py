from typing import (
    List,
    Any,
    Optional,
    Collection,
    TypeVar,
    Sequence,
    Union,
    Tuple,
    Callable,
    Dict,
)
import threading
import math
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np
from operator import methodcaller
import subprocess
from tqdm import tqdm
import json
import os
from Bio import SeqIO
import numba
from .tensor import collate_tensors
from .typed import PathLike
from .align import MSA
from .tokenization import Vocab
import re
import random
import pickle
import pandas as pd

T = TypeVar("T")


class ThreadsafeFile:
    def __init__(
        self,
        filepath: PathLike,
        open_func: Callable[[PathLike], T],
        close_func: Callable[[T], None] = methodcaller("close"),
    ):
        self._threadlocal = threading.local()
        self._filepath = filepath
        self._open_func = open_func
        self._close_func = close_func

    def __getattr__(self, name: str):
        return getattr(self.file, name)

    @property
    def file(self) -> T:
        if not hasattr(self._threadlocal, "file"):
            self._threadlocal.file = self._open_func(self._filepath)
        return self._threadlocal.file

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_threadlocal"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._threadlocal = threading.local()

    def __del__(self):
        if hasattr(self._threadlocal, "file"):
            self._close_func(self._threadlocal.file)
            del self._threadlocal.file


class SizedDataset(torch.utils.data.Dataset):
    def __init__(self, sizes: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self._sizes = sizes

    def __len__(self):
        return len(self.sizes)

    @property
    def sizes(self):
        return self._sizes


class CollatableDataset(torch.utils.data.Dataset):
    def collater(self, batch: List[Any]) -> Any:
        try:
            return torch.stack(batch, 0)
        except Exception:
            return batch


class CollatableVocabDataset(CollatableDataset):
    def __init__(self, vocab: Vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self.vocab = vocab


class BaseWrapperDataset(CollatableVocabDataset):
    """BaseWrapperDataset. Wraps an existing dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: CollatableVocabDataset):
        super().__init__(dataset.vocab)
        self.dataset = dataset

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class NPZDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        lazy: bool = False,
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.npz")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = sorted(file_list)
        self._lazy = lazy

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = np.load(str(self._file_list[index]))
        if not self._lazy:
            item = dict(item)
        return item

class ASADataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        lazy: bool = False,
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.asa")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)

                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .asa files found in {data_file}")

        self._file_list = sorted(file_list)
        self._lazy = lazy

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = pd.read_csv(self._file_list[index], delimiter=None, header=None, delim_whitespace=True,
                    skiprows=[0], comment='#').values[:, 2]
        item = item.astype(float)
        return item



class JsonDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of json files.
    Args:
        data_file (Union[str, Path]): Path to directory of lm_npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
        json_file Optional[Collection[str]]: Json file which contains
            labels for split_files.
    """

    def __init__(
        self,
        data_path: PathLike,
        split_files: Optional[Collection[str]] = None,
        json_file: Optional[Collection[str]] = None,
    ):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not data_path.is_dir():
            raise NotADirectoryError(data_path)

        split_files = list(split_files)
        if len(split_files) == 0:
            raise ValueError("Passed an empty split file set")
        json_file = open(os.path.join(data_path, json_file), "rb")
        self._json_file = json_file.read()
        self._split_files = split_files

    def __len__(self) -> int:
        return len(self._split_files)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = json.loads(self._json_file)[self._split_files[index]]
        return item

class PickleDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of pickle files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        lazy: bool = False,
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.pickle")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]

            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .pickle files found in {data_file}")

        self._file_list = sorted(file_list)
        self._lazy = lazy

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = pickle.load(
            open(str(self._file_list[index]), "rb")
        )
        if not self._lazy:
            item = dict(item)
        return item

class A3MDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of a3m files.
    Args:
        data_file (Union[str, Path]): Path to directory of a3m files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = None,
        sample_method: str = "fast",
    ):
        assert sample_method in ("fast", "best")
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.a2m")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .a3m files found in {data_file}")

        self._file_list = sorted(file_list)
        self._max_seqs_per_msa = max_seqs_per_msa
        self._sample_method = sample_method

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._max_seqs_per_msa == 1:
            seq = str(next(SeqIO.parse(self._file_list[index], "fasta")).seq)
            return seq
        else:
            msa = MSA.from_fasta(self._file_list[index])
            if self._max_seqs_per_msa is not None:
                msa = msa.select_diverse(
                    self._max_seqs_per_msa, method=self._sample_method
                )
            return msa


class FastaDataset(SizedDataset):
    """
    For loading protein sequence datasets in the common FASTA data format

    Modified from github.com/pytorch/fairseq.
    """

    def __init__(self, data_file: PathLike, cache_indices: bool = False):
        self.data_file = data_file
        self.file = ThreadsafeFile(data_file, open)
        self.cache = Path(f"{data_file}.idx.npy")
        if cache_indices:
            if self.cache.exists():
                self.offsets, sizes = np.load(str(self.cache))
            else:
                self.offsets, sizes = self._build_index()
                np.save(str(self.cache), np.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        super().__init__(sizes)

    def __getitem__(self, idx):
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        desc, *seq = data.split("\n", 1)
        seq = "".join(seq).split("\n")
        seq = "".join(seq).strip()
        seq = seq.upper()
        seq = re.sub(r"([a-z]|\.|\*)", "", seq)
        seq = re.sub(r"[T]", "U", seq)
        seq = re.sub(r"[RYKMSWBDHVN-]", "X", seq)
        # seq = self.rna_trans_protein(seq)
        return desc[1:], seq

    def __len__(self):
        return self.offsets.size

    def _build_index(self):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        bytes_offsets = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            "| grep --byte-offset '^>' -o | cut -d: -f1",
            shell=True,
        )
        fasta_lengths = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
            "'{print length($1)}'",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=" ")
        return bytes_np, sizes_np

    @classmethod
    def rna_trans_protein(cls, rna_seq):
        codonTable = {
            'AUA': 'I', 'AUC': 'I', 'AUU': 'I', 'AUG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T',
            'AAC': 'N', 'AAU': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGU': 'S', 'AGA': 'R', 'AGG': 'R',
            'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L',
            'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P',
            'CAC': 'H', 'CAU': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGU': 'R',
            'GUA': 'V', 'GUC': 'V', 'GUG': 'V', 'GUU': 'V',
            'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A',
            'GAC': 'D', 'GAU': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G',
            'UCA': 'S', 'UCC': 'S', 'UCG': 'S', 'UCU': 'S',
            'UUC': 'F', 'UUU': 'F', 'UUA': 'L', 'UUG': 'L',
            'UAC': 'Y', 'UAU': 'Y', 'UAA': 'X', 'UAG': 'X',
            'UGC': 'C', 'UGU': 'C', 'UGA': 'X', 'UGG': 'W',
        }
        proteinSeq = ""
        random_letters = ''.join(random.choices('ACGU', k=2))
        rna_seq += random_letters
        for codonStart in range(0, len(rna_seq) - 2, 1):
            codon = rna_seq[codonStart:codonStart + 3]
            if codon in codonTable:
                proteinSeq += codonTable[codon]
            else:
                proteinSeq += "X"

        return proteinSeq


class EncodedFastaDataset(CollatableVocabDataset, FastaDataset):
    def __init__(self, data_file: PathLike, vocab: Vocab):
        super().__init__(data_file=data_file, vocab=vocab, cache_indices=True)
        self._sizes += int(self.vocab.prepend_bos) + int(self.vocab.append_eos)

    def __getitem__(self, index: int) -> torch.Tensor:
        desc, seq = super().__getitem__(index)
        return torch.from_numpy(self.vocab.encode_single_sequence(seq))

    def collater(self, batch):
        return collate_tensors(batch, constant_value=self.vocab.pad_idx)


class MappingFastaDataset(FastaDataset):
    def __init__(self, data_file: PathLike, rna_to_protein: dict):
        super().__init__(data_file=data_file, cache_indices=True)
        self.mapping = rna_to_protein

    def __getitem__(self, index: int):
        desc, seq = super().__getitem__(index)
        mapped_seq = ''.join(self.mapping[char] if char in self.mapping else char for char in seq)
        return mapped_seq, seq




class MaxTokenBatch(object):
    def __init__(self, max_tokens: int, pad_idx: int):
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.items: List[torch.Tensor] = []
        self.sizes = None

    def can_add_item(self, item: torch.Tensor) -> bool:
        sizes = np.asarray(item.size())
        if self.sizes is not None:
            sizes = np.max([self.sizes, sizes], 0)
        total_tokens = (len(self.items) + 1) * sizes.prod()
        return total_tokens <= self.max_tokens

    def add_item(self, item: torch.Tensor):
        self.items.append(item)
        sizes = np.asarray(item.size())
        if self.sizes is None:
            self.sizes = sizes
        else:
            self.sizes = np.max([self.sizes, sizes], 0)
        if self.num_tokens > self.max_tokens:
            raise RuntimeError("Too many sequences in batch!")

    def finalize(self) -> torch.Tensor:
        return collate_tensors(self.items, constant_value=self.pad_idx)

    @property
    def num_tokens(self) -> int:
        if self.sizes is None:
            return 0
        else:
            return len(self.items) * self.sizes.prod()


BatchOrSequence = TypeVar("BatchOrSequence", MaxTokenBatch, Sequence[MaxTokenBatch])


class AutoBatchingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: CollatableVocabDataset, max_tokens: int, shuffle: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def maybe_make_and_add_batch(
        self,
        batch: Optional[BatchOrSequence],
        item: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Tuple[BatchOrSequence, bool]:
        if batch is None:
            if isinstance(item, torch.Tensor):
                batch = MaxTokenBatch(  # type: ignore
                    self.max_tokens, self.vocab.pad_idx
                )
            else:
                batch = [  # type: ignore
                    MaxTokenBatch(self.max_tokens, self.vocab.pad_idx) for _ in item
                ]

        if isinstance(batch, MaxTokenBatch):
            can_add = batch.can_add_item(item)  # type: ignore
            if can_add:
                batch.add_item(item)  # type: ignore
        else:
            can_add = batch[0].can_add_item(item[0])  # type: ignore
            if can_add:
                for b, i in zip(batch, item):  # type: ignore
                    b.add_item(i)
        return batch, can_add  # type: ignore

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            worker_rank = dist.get_rank()
        else:
            world_size = 1
            worker_rank = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            world_size *= worker_info.num_workers
            worker_rank = worker_rank * worker_rank.num_workers + worker_info.id

        chunk_size = math.ceil(len(indices) / world_size)
        indices = indices[chunk_size * worker_rank : chunk_size * (worker_rank + 1)]

        if self.shuffle:
            indices = np.random.permutation(indices)

        batch = None
        for idx in indices:
            items = self.dataset[idx]
            batch, added = self.maybe_make_and_add_batch(batch, items)
            if not added:
                if isinstance(batch, MaxTokenBatch):
                    yield batch.finalize()
                else:
                    yield type(items)(b.finalize() for b in batch)
                batch, added = self.maybe_make_and_add_batch(None, items)
                if not added:
                    breakpoint()
                assert added, "Item size too large to include!"
        if batch:
            if isinstance(batch, MaxTokenBatch):
                yield batch.finalize()
            else:
                yield type(items)(b.finalize() for b in batch)

    def collater(self, batch: List[Any]) -> Any:
        src = collate_tensors(
            [el[0] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        tgt = collate_tensors(
            [el[1] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        return src, tgt


@numba.njit
def batch_by_size(
    indices: np.ndarray, sizes: np.ndarray, max_tokens: int
) -> List[List[int]]:
    batches: List[List[int]] = []
    batch: List[int] = [0][:0]
    batch_size = 0
    for i in range(len(indices)):
        idx = indices[i]
        size = sizes[i]
        if size > max_tokens:
            raise RuntimeError("An item was too large to batch.")
        if size + batch_size > max_tokens:
            batches.append(batch)
            batch = [0][:0]
            batch_size = 0
        batch.append(idx)
        batch_size += size
    batches.append(batch)
    return batches


class BatchBySequenceLength(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: SizedDataset,
        max_tokens: int,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset)
        indices = np.argsort(dataset.sizes)
        sizes = dataset.sizes[indices]  # seq_len + 2 (prepend_bos, append_eos)
        batches = batch_by_size(indices, sizes, max_tokens)

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.batches = batches
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
        else:
            indices = list(range(len(self.batches)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == len(self)
        yield from (self.batches[idx] for idx in indices)

    def __len__(self):
        return math.ceil(len(self.batches) / self.num_replicas)

    @property
    def num_replicas(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    @property
    def total_size(self) -> int:
        return len(self) * self.num_replicas

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        else:
            return 0

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all
        replicas use a different random ordering for each epoch. Otherwise, the next
        iteration of this sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RandomCropDataset(BaseWrapperDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_seqlen: int):
        super().__init__(dataset)
        self.max_seqlen = max_seqlen
        self.num_special = int(self.vocab.prepend_bos) + int(self.vocab.append_eos)
        self.max_seqlen_no_special = self.max_seqlen - self.num_special
        self.sizes = np.minimum(self.sizes, max_seqlen)  # type: ignore

    def __getitem__(self, idx):
        item = self.dataset[idx]
        seqlen = item.size(-1)
        if seqlen > self.max_seqlen:
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, high_idx)
            end_idx = start_idx + self.max_seqlen_no_special
            item = torch.cat(
                [
                    item[..., :low_idx],
                    item[..., start_idx:end_idx],
                    item[..., high_idx:],
                ],
                -1,
            )
        return item

class RandomCropBPRNADataset(BaseWrapperDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_seqlen: int):
        super().__init__(dataset)
        self.max_seqlen = max_seqlen
        self.num_special = int(self.vocab.prepend_bos) + int(self.vocab.append_eos)
        self.max_seqlen_no_special = self.max_seqlen - self.num_special

    def __getitem__(self, idx):
        rnaid, tokens, contacts, missing_nt_index = self.dataset[idx]
        seqlen = tokens.size(-1)

        if seqlen > self.max_seqlen:
            # print(f"1:===={tokens.shape, contacts.shape}===")
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, high_idx - self.max_seqlen_no_special)
            end_idx = start_idx + self.max_seqlen_no_special
            # print(f"2:===={len(tokens[start_idx:end_idx]), start_idx, end_idx}===")
            tokens = torch.cat(
                [
                    tokens[:low_idx],
                    tokens[start_idx:end_idx],
                    tokens[high_idx:],
                ]
            )
            contacts = contacts[start_idx:end_idx, start_idx:end_idx]
            # print(f"****{seqlen, tokens.shape, contacts.shape}****")
        return rnaid, tokens, contacts, missing_nt_index


class SubsampleMSADataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        max_tokens: int,
        max_seqs: Optional[int] = None,
    ):
        super().__init__(dataset)
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs if max_seqs is not None else float("inf")

    def __getitem__(self, idx):
        msa = self.dataset[idx]

        num_alignments, seqlen = msa.size()
        max_alignments = self.max_tokens // seqlen
        max_alignments = min(self.max_seqs, max_alignments)
        if max_alignments < num_alignments:
            indices = np.random.randint(1, num_alignments, size=max_alignments - 1)
            indices = np.append(0, indices)
            msa = msa[indices]

        return msa


# class MaskedTokenWrapperDataset(BaseWrapperDataset):
#     def __init__(
#         self,
#         dataset: CollatableVocabDataset,
#         mask_prob: float = 0.15,
#         random_token_prob: float = 0.1,
#         leave_unmasked_prob: float = 0.1,
#         vocab_list: Optional[List[int]] = None,
#     ):
#         # TODO - add column masking?
#         # TODO - add collater
#         super().__init__(dataset)
#         assert 0 <= mask_prob <= 1
#         assert 0 <= random_token_prob <= 1
#         assert 0 <= leave_unmasked_prob <= 1
#
#         self._mask_prob = mask_prob
#         self._random_token_prob = random_token_prob
#         self._leave_unmasked_prob = leave_unmasked_prob
#         self.vocab_list = torch.tensor(vocab_list)
#
#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         masked_indices = torch.bernoulli(torch.full(item.shape, self.mask_prob)).bool()
#         masked_indices[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = False
#         tgt = item.masked_fill(~masked_indices, self.vocab.pad_idx)
#
#         mask_with_token = torch.bernoulli(
#             torch.full(item.shape, (1 - self._random_token_prob - self._leave_unmasked_prob))
#         ).bool() & masked_indices
#         src = item.masked_fill(mask_with_token, self.vocab.mask_idx)
#
#         mask_with_random = torch.bernoulli(torch.full(item.shape, 0.5)).bool() & masked_indices & ~mask_with_token
#         rand_tokens = self.vocab_list[torch.randint(0, len(self.vocab_list), item.shape)]
#         src[mask_with_random] = rand_tokens[mask_with_random]
#
#         return src, tgt
#     @property
#     def mask_prob(self) -> float:
#         return self._mask_prob
#
#     @property
#     def random_token_prob(self) -> float:
#         return self._random_token_prob
#
#     @property
#     def leave_unmasked_prob(self) -> float:
#         return self._leave_unmasked_prob
#
#     def collater(self, batch: List[Any]) -> Any:
#         src = collate_tensors(
#             [el[0] for el in batch],
#             constant_value=self.vocab.pad_idx,
#         )
#         tgt = collate_tensors(
#             [el[1] for el in batch],
#             constant_value=self.vocab.pad_idx,
#         )
#         return src, tgt


class MaskedTokenWrapperDataset(BaseWrapperDataset):
    def __init__(
            self,
            dataset: CollatableVocabDataset,
            mask_prob: float = 0.15,
            random_token_prob: float = 0.1,
            leave_unmasked_prob: float = 0.1,
    ):
        # TODO - add column masking?
        # TODO - add collater
        super().__init__(dataset)
        assert 0 <= mask_prob <= 1
        assert 0 <= random_token_prob <= 1
        assert 0 <= leave_unmasked_prob <= 1

        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob

    def __getitem__(self, idx):
        item = self.dataset[idx]
        random_probs = torch.rand_like(item, dtype=torch.float)
        random_probs[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = 1
        do_mask = random_probs < self.mask_prob

        tgt = item.masked_fill(~do_mask, self.vocab.pad_idx)
        mask_with_token = random_probs < (
                self.mask_prob * (1 - self.leave_unmasked_prob)
        )
        src = item.masked_fill(mask_with_token, self.vocab.mask_idx)
        mask_with_random = random_probs < (self.mask_prob * self.random_token_prob)
        # TODO - maybe prevent special tokens?
        rand_tokens = torch.randint_like(src, len(self.vocab))
        src[mask_with_random] = rand_tokens[mask_with_random]
        return src, tgt

    @property
    def mask_prob(self) -> float:
        return self._mask_prob

    @property
    def random_token_prob(self) -> float:
        return self._random_token_prob

    @property
    def leave_unmasked_prob(self) -> float:
        return self._leave_unmasked_prob

    def collater(self, batch: List[Any]) -> Any:
        src = collate_tensors(
            [el[0] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        tgt = collate_tensors(
            [el[1] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        return src, tgt


class KmerMaskedTokenWrapperDataset(BaseWrapperDataset):
    def __init__(
            self,
            dataset: CollatableVocabDataset,
            mask_prob: float = 0.15,
            random_token_prob: float = 0.1,
            leave_unmasked_prob: float = 0.1,
            kmer_ratio: float = 0.8
    ):
        super().__init__(dataset)
        assert 0 <= mask_prob <= 1
        assert 0 <= random_token_prob <= 1
        assert 0 <= leave_unmasked_prob <= 1
        assert 0 <= kmer_ratio <= 1

        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob
        self._kmer_ratio = kmer_ratio

    def __getitem__(self, idx):
        item = self.dataset[idx]
        random_probs = torch.rand_like(item, dtype=torch.float)
        random_probs[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = 1
        total_mask_count = int(self._mask_prob * len(item))
        kmer_mask_count = int(self._kmer_ratio * total_mask_count)
        single_mask_count = total_mask_count - kmer_mask_count

        masked_indices = set()


        def mask_kmer(start_idx, k):
            for i in range(start_idx, min(start_idx + k, len(item))):
                masked_indices.add(i)

        kmers_masked = 0
        while kmers_masked < kmer_mask_count:
            k = random.randint(3, 8)
            start_idx = random.randint(0, len(item) - k)
            if any(i in masked_indices for i in range(start_idx, start_idx + k)):
                continue
            mask_kmer(start_idx, k)
            kmers_masked += k


        singles_masked = 0
        while singles_masked < single_mask_count:
            idx = random.randint(0, len(item) - 1)
            if idx not in masked_indices:
                masked_indices.add(idx)
                singles_masked += 1


        do_mask = torch.zeros_like(item, dtype=torch.bool)
        for idx in masked_indices:
            do_mask[idx] = True

        tgt = item.masked_fill(~do_mask, self.vocab.pad_idx)
        src = item.masked_fill(do_mask, self.vocab.mask_idx)
        # mask_with_token = torch.rand_like(item, dtype=torch.float) < (
        #             self._mask_prob * (1 - self._leave_unmasked_prob))
        # src = item.masked_fill(do_mask & mask_with_token, self.vocab.mask_idx)
        # mask_with_random = torch.rand_like(item, dtype=torch.float) < (self._mask_prob * self._random_token_prob)
        #
        # rand_tokens = torch.randint_like(src, len(self.vocab))
        # src[mask_with_random] = rand_tokens[mask_with_random]

        return src, tgt

    @property
    def mask_prob(self) -> float:
        return self._mask_prob

    @property
    def random_token_prob(self) -> float:
        return self._random_token_prob

    @property
    def leave_unmasked_prob(self) -> float:
        return self._leave_unmasked_prob

    @property
    def kmer_ratio(self) -> float:
        return self._kmer_ratio

    def collater(self, batch: List[Any]) -> Any:
        src = collate_tensors(
            [el[0] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        tgt = collate_tensors(
            [el[1] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        return src, tgt

