# llava_next_processor.py
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from .base_processor import LMMPromptProcessor


try:
    # transformers>=4.37-ish 才有（不同版本可能有差异）
    from transformers import LlavaNextProcessor  # type: ignore
except Exception:
    LlavaNextProcessor = None  # type: ignore

from transformers import AutoProcessor


PromptItem = Union[str, "PIL.Image.Image"]  # 也支持 str=path/url（会被 is_img 解析）
Prompt = Sequence[PromptItem]
BatchPrompts = Union[Prompt, Sequence[Prompt]]


class LlavaNextPromptProcessor(LMMPromptProcessor):
    """
    约定：
      - batch_prompts: List[List[Union[str, PIL.Image, path, url]]]
        图文交错；遇到图片就在 text 中插入 <image>，并把图片加入 images list。
      - 输出：HF processor 的 BatchFeature，通常包含 input_ids/attention_mask/pixel_values（以及部分模型的 image_sizes 等）。
    """

    def __init__(
        self,
        model_name_or_path: str,
        image_token: str = "<image>",
        **processor_kwargs: Any,
    ):
        # 优先用专用 Processor；不行就 fallback AutoProcessor（适配不同 transformers 版本）
        if LlavaNextProcessor is not None:
            self.processor = LlavaNextProcessor.from_pretrained(
                model_name_or_path, **processor_kwargs
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, **processor_kwargs
            )

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            # 极少数旧接口可能叫 feature_extractor
            image_processor = getattr(self.processor, "feature_extractor", None)

        super().__init__(self.processor.tokenizer, image_processor)
        self.image_token = image_token

    def prepare_input(
        self,
        batch_prompts: BatchPrompts,
        padding: str = "longest",
        truncation: Optional[bool] = None,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
        add_eos_token: bool = False,
        debug: bool = False,
        # 可选：如果你希望强行包一层最常见的 LLaVA 对话格式（有些评测需要）
        wrap_user_assistant: bool = False,
    ):
        # 和 OpenFlamingo 的写法一致：如果用户传的是单条 prompt（不是 batch），自动包一层 batch :contentReference[oaicite:2]{index=2}
        if not any(isinstance(i, (list, tuple)) for i in batch_prompts):  # type: ignore
            batch_prompts = [batch_prompts]  # type: ignore

        batch_text_inputs: List[str] = []
        batch_image_inputs: List[List[Any]] = []

        for prompts in batch_prompts:  # type: ignore
            image_inputs: List[Any] = []
            text_inputs: str = ""

            for item in prompts:
                item_is_img = self.is_img(item)
                if item_is_img is None:
                    # 非图像：当作文本拼接
                    text_inputs += str(item).strip(" ")
                else:
                    # 图像：插入占位 token，并把图加入 images
                    image_inputs.append(item_is_img)
                    text_inputs += self.image_token

            if wrap_user_assistant:
                # 这是最朴素的包法；如果你自己已经在外面拼好了聊天模板，就把这个关掉
                text_inputs = f"USER: {text_inputs}\nASSISTANT:"

            if add_eos_token:
                # tokenizer.eos_token 更稳妥（有些模型不是 </s>）
                eos = getattr(self.tokenizer, "eos_token", None) or "</s>"
                text_inputs += eos

            if debug:
                print("==== LlavaNextPromptProcessor sample ====")
                print(text_inputs)
                print(f"num_images={len(image_inputs)}")

            batch_text_inputs.append(text_inputs)
            batch_image_inputs.append(image_inputs)

        # 有些 HF processor 在“全 batch 都没图”时，传 images=[] 可能会不开心；这里做个兼容
        use_images = any(len(x) > 0 for x in batch_image_inputs)

        processor_kwargs = dict(
            text=batch_text_inputs,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        if use_images:
            processor_kwargs["images"] = batch_image_inputs

        return self.processor(**processor_kwargs)
