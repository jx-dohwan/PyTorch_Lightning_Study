# https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py
import logging as log
from argparse import ArguemtParser, Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, 
from transformers import AutoModel

from tokenizer import Tokenizer
from utils import mask_fill

class Classifier(pl.Lightningmodule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class DataModule(pl.LightningDataMudule):
        def __init__(self, classifier_instance):
            super().__init__()
            self.hparams = classifier_instance.hparams
            self.classifier = classifier_instance
            # Label Encoder
            self.label_encoder = LabelEncoder(
                pd.read_csv(self.hparams.train.csv).label.astype(str).unique().tolist(),
                reserved_labels=[],
            )
            self.label_encoder.unknown_index = None

        def read_csv(self, path:str) -> list:
            """Reads a comma separated value file.
            :param path:path to a csv file.
            :return:List of records as dictionaries
            """
            df = pd.read_csv(path)
            df = df[['text','label']]
            df['text'] = df['text'].astype(str)
            df['label'] = df['label'].astype(str)
            return df.to_dict('records')

        def train_dataloader(self) -> DataLoader:
            """Function that loads the train set."""
            self._train_dataset = self.read_csv(self.hparams.train_csv)
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            """Function that loads the validation set."""
            self._dev_dataset = self.read_csv(self.hparams.dev_csv)
            return DataLoader(
                dataset = self._test_dataset,
                batch_size = self.hparams.batch_size,
                collate_fn=self.classifier.prepare_smaple,
                num_workers=self.hparams.loader_workers,
            )
    def __init__(self, hparams:Namespace) -> None:
        super(Classifier, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # Build Data Module
        self.data = self.DataModule(self)

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """Init BERT model + tokenizer + classification head."""
        self.bert = AutoModel.from_pretrained(
            self.hparams.encoder_model, output_hidden_states=True
        )
        # set the number of features our encoder model will return...
        self.encoder_features = self.bert.config.hidden_size

        # Tokenizer
        self.tokenizer = Tokenizer(self.hparams.encoder_model)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),
        )
    
    def __build_loss(self):
        """Initializes the loss function/s."""
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out['logits'].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample['predicted_label'] = predicted_labels[0]
            
        return sample

