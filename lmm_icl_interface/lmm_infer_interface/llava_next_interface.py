# lmm_icl_interface/interfaces/llava_next_interface.py
from __future__ import annotations

import torch
from loguru import logger

from .base_interface import LMMInterface
from lmm_icl_interface.lmm_processor import LlavaNextPromptProcessor


class LlavaNextInterface(LMMInterface):
    def __init__(
        self,
        model_name_or_path: str,
        precision,
        model_device,
        prompt_manager,
        instruction: str,
        image_field: str,
        label_field: str,
        processor_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ):
        """
        参数风格对齐 Idefics2Interface：model_name_or_path / precision / model_device / prompt_manager / instruction / image_field / label_field
        :contentReference[oaicite:5]{index=5}
        """
        super().__init__(
            precision=precision,
            input_ids_field_name="input_ids",
            prompt_manager=prompt_manager,
            instruction=instruction,
            label_field=label_field,
            image_field=image_field,
        )

        processor_kwargs = processor_kwargs or {}
        model_kwargs = model_kwargs or {}

        # 1) Processor
        self.processor = LlavaNextPromptProcessor(model_name_or_path, **processor_kwargs)

        # 2) Model（transformers 内置 LLaVA-NeXT）
        from transformers import LlavaNextForConditionalGeneration

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=self.data_type,
            **model_kwargs,
        ).to(model_device)
        self.model.eval()

        # 3) Tokenizer & image processor
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        # 左 padding：你其它接口也这么设，方便 generation / ppl 对齐 :contentReference[oaicite:6]{index=6}
        self.tokenizer.padding_side = "left"

        # LLaVA 系很多 tokenizer 默认没 pad_token，用 eos 顶一下更稳（否则 ppl 的 ignore_index 会有坑）
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                logger.warning("Tokenizer has no pad_token_id and no eos_token; please set pad_token manually.")
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id

        # 一些下游逻辑可能会用到（你 Idefics/Idefics2 也保存了这些）:contentReference[oaicite:7]{index=7}
        self.fake_token = "<fake_token_around_image>"
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # generation config 也同步一下 pad_token_id，避免 generate 报 warning
        try:
            if getattr(self.model.generation_config, "pad_token_id", None) is None:
                self.model.generation_config.pad_token_id = self.pad_token_id
        except Exception:
            pass
