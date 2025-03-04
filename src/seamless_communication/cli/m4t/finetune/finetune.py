# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import wandb
from typing import Any, Dict


import torch
from datasets import load_dataset

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.unity import UnitYModel

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune")


def load_checkpoint(model: UnitYModel, path: str, device = torch.device("cpu")) -> None:
    saved_model = torch.load(path, map_location=device)["model"]
    saved_model = { k.replace("model.", ""): v for k, v in saved_model.items() }

    def _select_keys(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}

    model.speech_encoder_frontend.load_state_dict(_select_keys(saved_model, "model.speech_encoder_frontend."))
    model.speech_encoder.load_state_dict(_select_keys(saved_model, "model.speech_encoder."))

    assert model.text_decoder_frontend is not None
    model.text_decoder_frontend.load_state_dict(_select_keys(saved_model, "model.text_decoder_frontend."))

    assert model.text_decoder is not None
    model.text_decoder.load_state_dict(_select_keys(saved_model, "model.text_decoder."))

    assert model.final_proj is not None
    model.final_proj.load_state_dict(_select_keys(saved_model, "model.final_proj."))

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Load model checkpoint for further finetuning",
        default=None
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help=("Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        required=False,
        default=None,
        # TODO: better description
        help=("A list of modules to freeze in the model. If empty, everything will be trained."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help=("Activate usage of wandb to report training metrics. Log into wandb by yourself via console.")
    )
    parser.add_argument(
        '--is_source_cvss',
        action="store_true",
        help="If the source dataset is CVSS, it will load the audio_arrays directly"
    )
    parser.add_argument(
        '--grad_accum_steps',
        type=int,
        default=4,
        help="Specify the number of gradient accumulation steps"
    )

    return parser


def main() -> None:
    args = init_parser().parse_args()
    
    dist_utils.init_distributed([logger, trainer.logger])
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=args.mode,
        save_model_path=args.save_model_to,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        grad_accum_steps=4
    )
    
    logger.info(f"Finetune Params: {finetune_params}")
    
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float16)
    
    print("Loading checkpoint...")
    if args.checkpoint is not None: 
        st_dict = torch.load(args.checkpoint)
    
        # Get model name from state dict or use default
        model_name = st_dict.get('model_name', 'seamlessM4T_large')
        print(f"Using model: {model_name}")
    else:
        model_name = 'seamlessM4T_large'
    if args.checkpoint is not None:
        # Need to handle the module.model prefix in state dict keys
        print("Adapting checkpoint state dict...")
        # Create a new state dict with corrected keys
        new_state_dict = {}
        for key, value in st_dict['model'].items():
            # Remove 'module.model.' prefix
            if key.startswith('module.model.'):
                new_key = key[len('module.model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the adapted state dict
        print("Loading model weights...")
        model.load_state_dict(new_state_dict, strict=False)
        
    
    print(model)
    assert model.target_vocab_info == text_tokenizer.vocab_info
    
    if (
        finetune_params.finetune_mode == trainer.FinetuneMode.SPEECH_TO_TEXT
        and model.t2u_model is not None
    ):
        model.t2u_model = None
    
    if model.text_encoder is not None:
        model.text_encoder = None
    
    # Put model on selected device
    model = model.to(finetune_params.device)

    if args.use_wandb:
        wandb.init(
        project="seamless-m4t-medium-finetune",
        config={
            "model": "seamless-m4t-medium",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs":args.max_epochs,
            "dataset":"CVSS"
        }
    )

    cvss_EN_train_dataset, cvss_EN_eval_dataset = None, None

    if args.is_source_cvss:
        cvss_EN_train_dataset = load_dataset(
            'ebellob/cvss-c-fleurs-format-target',
            split='train'
        )
        # cvss_ES_train_dataset = load_dataset(
        #     'ebellob/cvss-c-fleurs-format-source',
        #     split='train'
        # )
        cvss_EN_eval_dataset = load_dataset(
            'ebellob/cvss-c-fleurs-format-target',
            split='validation'
        )
        # cvss_ES_eval_dataset = load_dataset(
        #     'ebellob/cvss-c-fleurs-format-source',
        #     split='validation'
        # )

    # TODO: delete unused params to reduce GPU memory consumption
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=15.0,
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.train_dataset,
        cvss_dataset=cvss_EN_train_dataset,

        max_src_tokens_per_batch=args.max_src_tokens)
    
    eval_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=75.0,
            float_dtype=finetune_params.float_dtype,
        ),
        cvss_dataset=cvss_EN_eval_dataset,
        dataset_manifest_path=args.eval_dataset)
    
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=args.freeze_layers,
        use_wandb=args.use_wandb)
        
    
    finetune.run()


if __name__ == "__main__":
    main()
