# from __future__ import annotations

# import json
# import pathlib
# import copy

# from typing import Any
# from transformers.configuration_utils import PretrainedConfig


# class ModelConfig(PretrainedConfig):

#     # def __init__(self: ModelConfig, config_file: pathlib.Path | str | None = None, **kwargs):
#     #     """
#     #     """
#     #     super().__init__(**kwargs)
#     #     if config_file is None:
#     #         self.attention_probs_dropout_prob: float = 0.1
#     #         self.hidden_dropout_prob = 0.1
#     #         self.hidden_size = 768
#     #         self.intermediate_size = 1280
#     #         self.max_sequence_length = 512
#     #         self.position_bucket_size = 32
#     #         self.num_attention_heads = 6
#     #         self.num_layers = 12
#     #         self.vocab_size = 16384
#     #         self.layer_norm_eps = 1e-7
#     #     else:
#     #         if config_file == "str":
#     #             config_file = pathlib.Path(config_file)

#     #         config: dict[str, Any] = json.load(config_file.open("r"))

#     #         for key, value in config.items():
#     #             setattr(self, key, value)
# # Correct, standard implementation for your __init__ method
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.attention_probs_dropout_prob = kwargs.get("attention_probs_dropout_prob", 0.1)
#         self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)
#         self.hidden_size = kwargs.get("hidden_size", 768) # Default can be your most common size
#         self.intermediate_size = kwargs.get("intermediate_size", 2560)
#         self.max_position_embeddings = kwargs.get("max_position_embeddings", 512)
#         self.position_bucket_size = kwargs.get("position_bucket_size", 32)
#         self.num_attention_heads = kwargs.get("num_attention_heads", 12)
#         self.num_layers = kwargs.get("num_layers", 12)
#         self.vocab_size = kwargs.get("vocab_size", 16384)
#         self.layer_norm_eps = kwargs.get("layer_norm_eps", 1.0e-5)
#     def __repr__(self) -> str:
#         return str(self.to_json_string())

#     def to_dict(self) -> dict[str, Any]:
#         """Serializes this instance to a Python dictionary."""
#         output: dict[str, Any] = copy.deepcopy(self.__dict__)
#         return output

#     def to_json_string(self) -> str:
#         """Serializes this instance to a JSON string."""
#         return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

#     def to_json_file(self, json_file_path: pathlib.Path | str) -> None:
#         """Save this instance to a json file."""
#         if isinstance(json_file_path, str):
#             json_file_path: pathlib.Path = pathlib.Path(json_file_path)
#         with json_file_path.open("w", encoding='utf-8') as writer:
#             writer.write(self.to_json_string())
from __future__ import annotations
from typing import Any
from transformers.configuration_utils import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "gpt-bert"  # important for HF

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Core hparams (defaults; override via config.json or kwargs)
        self.attention_probs_dropout_prob = kwargs.get("attention_probs_dropout_prob", 0.1)
        self.hidden_dropout_prob          = kwargs.get("hidden_dropout_prob", 0.1)
        self.hidden_size                  = kwargs.get("hidden_size", 768)
        self.intermediate_size            = kwargs.get("intermediate_size", 2560)
        self.max_position_embeddings      = kwargs.get("max_position_embeddings", 512)
        self.position_bucket_size         = kwargs.get("position_bucket_size", 32)
        self.num_layers                   = kwargs.get("num_layers", kwargs.get("num_hidden_layers", 12))
        self.num_hidden_layers            = kwargs.get("num_hidden_layers", self.num_layers)  # alias both ways
        self.num_attention_heads          = kwargs.get("num_attention_heads", 12)
        self.vocab_size                   = kwargs.get("vocab_size", 16384)
        self.layer_norm_eps               = kwargs.get("layer_norm_eps", 1.0e-5)

        # Tell HF which classes to import when trust_remote_code=True
        self.auto_map = kwargs.get(
            "auto_map",
            {
                "AutoConfig": "configuration_gpt_bert.ModelConfig",
                "AutoModel": "modeling_gpt_bert.GPTBERT",
                "AutoModelForCausalLM": "modeling_gpt_bert.GPTBERTForCausalLM",
                "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBERTForMaskedLM",
            },
        )

    # DO NOT override to_dict / to_json_string / to_json_file
    # If you really want to, mirror the signatures:
    # def to_json_string(self, use_diff: bool = True) -> str:
    #     return super().to_json_string(use_diff=use_diff)
    # def to_json_file(self, json_file_path, use_diff: bool = True) -> None:
    #     return super().to_json_file(json_file_path, use_diff=use_diff)
