# -*- coding: utf-8 -*-
"""Lazy-loaded Qwen3-VL backend for the VLM explanation V2 protocol."""

from __future__ import annotations

from typing import Mapping, Sequence

from .config_v2 import validate_vlm_config_v2


class Qwen3VLBackendV2:
    """Qwen3-VL inference backend using the versioned V2 config contract."""

    def __init__(self, *, model, processor, torch_module, vision_config, model_id: str):
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.vision_config = dict(vision_config)
        self.model_id = model_id

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "Qwen3VLBackendV2":
        normalized = validate_vlm_config_v2(config)
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-vlm.txt before loading Qwen3-VL"
            ) from error

        if normalized["model"]["require_cuda"] and not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3-VL-4B canonical path requires a CUDA GPU; select a Colab GPU runtime"
            )

        model_id = normalized["model"]["id"]
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=normalized["model"]["device_map"],
            low_cpu_mem_usage=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return cls(
            model=model,
            processor=processor,
            torch_module=torch,
            vision_config=normalized["vision"],
            model_id=model_id,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, object]],
        generation: Mapping[str, object],
    ) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise RuntimeError("qwen-vl-utils==0.0.14 is required") from error

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        images, videos = process_vision_info(
            messages,
            image_patch_size=int(self.vision_config["image_patch_size"]),
        )
        inputs = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            do_resize=False,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **dict(generation))
        prompt_length = int(inputs["input_ids"].shape[-1])
        generated = generated[:, prompt_length:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
