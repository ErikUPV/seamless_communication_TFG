# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import logging
import os
from abc import abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset, load_from_disk

from .datatypes import LangPairSample, MultimodalSample

logger = logging.getLogger(__name__)


class SpeechTokenizer:
    @abstractmethod
    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...


class Speech2SpeechFleursDatasetBuilder:
    """Assembles speech2speech dataset from google/fleurs on HuggingFace"""

    #DATASET_NAME = "google/fleurs"

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        split: str = "test",
        dataset_name: str = 'google/fleurs',
        skip_source_audio: bool = True,
        skip_target_audio: bool = True,
        audio_dtype: torch.dtype = torch.float32,
        dataset_cache_dir: Optional[str] = None,
        speech_tokenizer: Optional[SpeechTokenizer] = None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.dataset_cache_dir = dataset_cache_dir
        self.audio_dtype = audio_dtype
        self.skip_source_audio = skip_source_audio
        self.skip_target_audio = skip_target_audio
        self.speech_tokenizer = speech_tokenizer
        self.dataset_name = dataset_name

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        should_skip_audio = (
            lang == self.target_lang
            and self.skip_target_audio
            or lang == self.source_lang
            and self.skip_source_audio
            or waveform_npy is None
        )
        if not should_skip_audio:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
        else:
            waveform = None
        if self.speech_tokenizer is not None and not should_skip_audio:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )

    def iterate_lang_audio_samples(self, part: str, lang: str, is_cvss: bool = False) -> Iterable[MultimodalSample]:
        cf = DownloadConfig(resume_download=True, num_proc=24)

        if self.dataset_name == 'google/fleurs':
            ds = load_dataset(
                'google/fleurs',
                lang,
                split=self.split,
                cache_dir=self.dataset_cache_dir,
                streaming=False,
                trust_remote_code=True,
            )
        if is_cvss and part == 'source':
#            ds = load_from_disk('~/s2st/fleurs_source')[self.split]
            ds = load_dataset(
                'ebellob/cvss-c-fleurs-format-source',
                split=self.split,
                streaming=False,
                trust_remote_code=True,
                num_proc=12
            )
        elif is_cvss and part == 'target':
#            ds = load_from_disk('~/s2st/fleurs_target')[self.split]
            ds = load_dataset(
                'ebellob/cvss-c-fleurs-format-target',
                split=self.split,
                streaming=False,
                trust_remote_code=True,
                num_proc=12
            )
        for item in ds:
            #print(ds)
            audio_path = os.path.join(
                os.path.dirname(item["path"]), item["audio"]["path"]
            )
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                item["id"],
                audio_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"],
                item["transcription"],
            )

            #print(f"The shape of the waveform is {waveform.shape}")

            #print(f"Waveform: {waveform}")
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=np.array(waveform),
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self, is_cvss=False) -> Iterable[LangPairSample]:
        logger.info(f"Loading {self.target_lang} samples")
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang, is_cvss=is_cvss, part='target')
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang, is_cvss=is_cvss, part='source')
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])


class Speech2TextGigaspeechDatasetBuilder:
    """ Assembles speech2speech dataset from google/fleurs on HuggingFace.
        This dataset requires signing an license agreement and using an auth token.
    """

    DATASET_NAME = "speechcolab/gigaspeech"

    def __init__(
        self,
        auth_token: str,
        split: str = "test",
        skip_source_audio: bool = True,
        skip_target_audio: bool = True,
        audio_dtype: torch.dtype = torch.float32,
        dataset_cache_dir: Optional[str] = None,
        speech_tokenizer: Optional[SpeechTokenizer] = None,
    ):
        self.auth_token = auth_token
        self.split = split
        self.dataset_cache_dir = dataset_cache_dir
        self.audio_dtype = audio_dtype
        self.skip_source_audio = skip_source_audio
        self.skip_target_audio = skip_target_audio
        self.speech_tokenizer = speech_tokenizer

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        if waveform_npy is not None:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
        else:
            waveform = None
        if self.speech_tokenizer is not None and waveform_npy is not None:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )

    def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:
        ds = load_dataset(
            self.DATASET_NAME,
            lang,
            split=self.split,
            cache_dir=self.dataset_cache_dir,
            streaming=False,
            trust_remote_code=True,
        )
        for item in ds:
            audio_path = os.path.join(
                os.path.dirname(item["path"]), item["audio"]["path"]
            )
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                item["id"],
                audio_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"],
                item["transcription"],
            )
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        logger.info(f"Loading {self.target_lang} samples")
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])
