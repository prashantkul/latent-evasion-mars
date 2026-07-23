import torch

from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix


class Qwen35_9b(LanguageModel):
    """A class to manage the 'Qwen/Qwen3.5-9B' model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        device="cuda",
        system_prompt=None,
        quantization_config=None,
        torch_dtype=None,
    ):
        super().__init__(model_name, system_prompt, device, quantization_config, torch_dtype)
        self.hidden_dimension = self.model.config.hidden_size
        self.num_layer = len(self._get_transformer_layers())
        self.refusal_token_id = [40]

    def _get_prompt(self, prompt="", tools=None, enable_thinking=None):
        if getattr(self.tokenizer, "chat_template", None):
            messages = []
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            # Non-agentic default is unchanged: with tools=None / enable_thinking=None neither
            # kwarg is forwarded, so the template behaves exactly as before. Agentic extraction
            # opts in by passing tools (OpenAI-style schemas render into context so the
            # post-instruction read matches the agentic distribution) and enable_thinking=False
            # (neutralises the reasoning block to match the thinking-off baseline).
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if tools is not None:
                template_kwargs["tools"] = tools
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

        raise NotImplementedError("Prompt construction for non-chat-template tokenizers is not implemented.")

    def _get_transformer_layers(self):
        return self.model.model.layers

    def _get_eoi_toks(self):
        return [self.tokenizer.eos_token_id]

    def _get_end_of_reasoning_toks(self):
        return self.tokenizer.encode("</think>\n\n", add_special_tokens=False)

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(
            self.model.model.embed_tokens.weight.data,
            direction,
        )

        for block in self.model.model.layers:
            layer_type = getattr(block, "layer_type", None)
            if layer_type == "linear_attention":
                attn_out_proj = block.linear_attn.out_proj
            else:
                attn_out_proj = block.self_attn.o_proj

            attn_out_proj.weight.data = get_ablated_matrix(
                attn_out_proj.weight.data.T,
                direction,
            ).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(
                block.mlp.down_proj.weight.data.T,
                direction,
            ).T

        print("✓ Weights ablated.")


class Qwen35_27b(Qwen35_9b):
    """The 'Qwen/Qwen3.5-27B' dense model (64 layers, hidden 5120). Inherits the
    tools-aware, thinking-off prompt builder from Qwen35_9b."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-27B",
        device="cuda",
        system_prompt=None,
        quantization_config=None,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            system_prompt=system_prompt,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
