from typing import Optional, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import sklearn.linear_model
from tqdm import tqdm
from dataclasses import dataclass

from evo.tokenization import Vocab
from evo.metrics import lmdata_compute_precisions
from evo.tensor import symmetrize, apc

from modules import (
    TransformerLayer,
    PKMLayer,
    ContactPredictionHead,
    LearnedPositionalEmbedding,
    RobertaLMHead,
)

from rna_esm.modules import (
    ESM1bLayerNorm,
    RobertaLMHead as esm2_RobertaLMHead,
    TransformerLayer as esm2_TransformerLayer,
)
from product_key_memory import PKM

import lr_schedulers
from dataset import TRRosettaContactDataset
from pathlib import Path

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
    resume_from_checkpoint: Optional[str] = None
    val_check_interval: int = 1000


class BaseProteinModel(pl.LightningModule, ABC):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.optimizer_config = optimizer_config
        self.contact_train_data = contact_train_data

    @abstractmethod
    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        return NotImplemented

    @abstractmethod
    def get_sequence_attention(self, tokens):
        return NotImplemented

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def on_validation_epoch_start(self):
        self.train_contact_regression()

    def train_contact_regression(self, verbose=False):
        data = self.contact_train_data
        if data is None:
            raise RuntimeError(
                "Cannot train regression without trRosetta contact training set."
            )
        X = []
        y = []
        with torch.no_grad():
            iterable = data if not verbose else tqdm(data)
            for rnaid, tokens,contacts,missing_nt_index in iterable:
                tokens = tokens.unsqueeze(0)
                attentions = self.get_sequence_attention(tokens)
                start_idx = int(self.vocab.prepend_bos)
                end_idx = attentions.size(-1) - int(self.vocab.append_eos)
                attentions = attentions[..., start_idx:end_idx, start_idx:end_idx]
                seqlen = attentions.size(-1)
                attentions = symmetrize(attentions)
                attentions = apc(attentions)
                attentions = attentions.view(-1, seqlen, seqlen).cpu().numpy()

                sep = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
                mask = sep >= 6
                if len(missing_nt_index) > 0:
                    for i in missing_nt_index:
                        mask[i, :] = False
                        mask[:, i] = False

                attentions = attentions[:, mask]
                attentions[np.isnan(attentions)] = 0
                contacts = contacts[mask]
                X.append(attentions.T)
                y.append(contacts)

        X = np.concatenate(X, 0)
        y = np.concatenate(y, 0)

        clf = sklearn.linear_model.LogisticRegression(
            penalty="l1",
            C=0.15,
            solver="liblinear",
            verbose=verbose,
            random_state=0,
        )
        clf.fit(X, y)

        self.contact_head.regression.load_state_dict(
            {
                "weight": torch.from_numpy(clf.coef_),
                "bias": torch.from_numpy(clf.intercept_),
            }
        )

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src)["logits"]
        valid_mask = tgt != self.vocab.pad_idx

        logits = logits[valid_mask]
        tgt = tgt[valid_mask]
        loss = nn.CrossEntropyLoss(reduction="none")(logits, tgt)
        perplexity = loss.float().exp().mean()
        loss = loss.mean()

        self.log("train/loss", loss)
        self.log("train/perplexity", perplexity, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        predictions = self.predict_contacts(batch["src_tokens"])

        result = {
            'rna_id': batch["rna_id"],
            'predictions': predictions,
            'tgt': batch["tgt"],
            'missing_nt_index': batch["missing_nt_index"],
        }

        return result

    def validation_epoch_end(self, validation_step_outputs):

        metrics = lmdata_compute_precisions(
            validation_step_outputs,
            minsep=0,
            step=0.001,
        )

        for key, value in metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True)


    def configure_optimizers(self):
        no_decay = ["norm", "LayerNorm"]

        pkm_params = []
        for module in self.modules():
            if isinstance(module, PKM):
                pkm_params.append(module.values.weight)
        pkm_paramset = set(pkm_params)

        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if param in pkm_paramset:
                continue

            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.optimizer_config.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
            {
                "params": pkm_params,
                "weight_decay": 0.0,
                "lr": 4 * self.optimizer_config.learning_rate,
            },
        ]

        if self.optimizer_config.name == "adam":
            optimizer_type = torch.optim.AdamW
        elif self.optimizer_config.name == "lamb":
            try:
                from apex.optimizers import FusedLAMB
            except ImportError:
                raise ImportError("Apex must be installed to use FusedLAMB optimizer.")
            optimizer_type = FusedLAMB
        optimizer = optimizer_type(
            optimizer_grouped_parameters,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.adam_betas,
        )
        scheduler = lr_schedulers.get(self.optimizer_config.lr_scheduler)(
            optimizer,
            self.optimizer_config.warmup_steps,
            self.optimizer_config.max_steps,
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]



class ESM2(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        model_config: TransformerConfig = TransformerConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
        token_dropout: bool = True,
    ):
        super().__init__(
            vocab=vocab,
            optimizer_config=optimizer_config,
            contact_train_data=contact_train_data,
        )
        self.model_config = model_config
        self.token_dropout = token_dropout
        self._init_submodules()

    def _init_submodules(self):
        config = self.model_config
        self.embed_scale = 1

        self.embed_tokens = nn.Embedding(
            len(self.vocab),
            config.embed_dim,
            padding_idx=self.vocab.pad_idx,
        )

        self.layers = nn.ModuleList(
            [
                esm2_TransformerLayer(
                    config.embed_dim,
                    4 * config.embed_dim,
                    config.num_attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    attention_type="standard",
                    performer_attention_features=256,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            config.num_layers * config.num_attention_heads,
            self.vocab.prepend_bos,
            self.vocab.append_eos,
            eos_idx=self.vocab.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(config.embed_dim)

        self.lm_head = esm2_RobertaLMHead(
            embed_dim=config.embed_dim,
            output_dim=len(self.vocab),
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)    # B, T, C

        if self.token_dropout:
            x.masked_fill_((tokens == self.vocab.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.vocab.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)    B；batch_size, T:target_length, E: embedding_dim
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def get_sequence_attention(self, tokens):
        return self(tokens.to(device=self.device), need_head_weights=True)["attentions"]



class ESM1b(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        model_config: TransformerConfig = TransformerConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
    ):
        super().__init__(
            vocab=vocab,
            optimizer_config=optimizer_config,
            contact_train_data=contact_train_data,
        )
        self.model_config = model_config

        self.embed_tokens = self.build_embedding()
        self.dropout_layer = nn.Dropout(model_config.layer.dropout)

        self.layers = nn.ModuleList([])
        for i in range(self.model_config.num_layers):
            if i in self.model_config.pkm_layers:
                layer: Union[TransformerLayer, PKMLayer] = self.build_pkm_layer()
            else:
                layer = self.build_transformer_layer()
            self.layers.append(layer)

        self.embed_positions = LearnedPositionalEmbedding(
            model_config.max_seqlen,
            self.model_config.layer.embed_dim,
            vocab.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(self.model_config.layer.embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(self.model_config.layer.embed_dim)
        self.lm_head = self.build_lm_head(weight=self.embed_tokens.weight)
        self.contact_head = self.build_contact_head()

        self.init_weights()

    def build_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            len(self.vocab),
            self.model_config.layer.embed_dim,
            padding_idx=self.vocab.pad_idx,
        )

    def build_transformer_layer(self) -> TransformerLayer:
        config = self.model_config.layer
        return TransformerLayer(
            embed_dim=config.embed_dim,
            ffn_embed_dim=4 * config.embed_dim,
            attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            attention_type=config.attention_type,
            performer_attention_features=config.performer_attention_features,
        )

    def build_pkm_layer(self) -> PKMLayer:
        config = self.model_config.pkm
        return PKMLayer(
            embed_dim=config.embed_dim,
            ffn_embed_dim=4 * config.embed_dim,
            attention_heads=config.num_attention_heads,
            pkm_attention_heads=config.pkm_attention_heads,
            pkm_dim_head=config.embed_dim // config.num_attention_heads,
            num_product_keys=config.num_product_keys,
            pkm_topk=config.pkm_topk,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            attention_type=config.attention_type,
            performer_attention_features=config.performer_attention_features,
        )

    def build_lm_head(self, weight: torch.Tensor) -> RobertaLMHead:
        return RobertaLMHead(
            embed_dim=self.model_config.layer.embed_dim,
            output_dim=len(self.vocab),
            weight=weight,
        )

    def build_contact_head(self) -> ContactPredictionHead:
        contact_head = ContactPredictionHead(
            self.model_config.num_layers * self.model_config.layer.num_attention_heads,
            self.vocab.prepend_bos,
            self.vocab.append_eos,
            eos_idx=self.vocab.eos_idx,
        )
        contact_head.requires_grad_(False)
        return contact_head

    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        if return_contacts:
            need_head_weights = True
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, T

        x = self.embed_tokens(tokens)

        x = x + self.embed_positions(tokens)

        x = self.emb_layer_norm_before(x)
        x = self.dropout_layer(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attentions = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attentions.append(attn.transpose(1, 0))
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attentions, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(
                    2
                )
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def get_sequence_attention(self, tokens):
        return self(tokens.to(device=self.device), need_head_weights=True)["attentions"]