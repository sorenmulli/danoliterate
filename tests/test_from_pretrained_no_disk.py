import torch
from transformers import AutoModelForCausalLM

from danoliterate.modeling.load_model import from_pretrained_hf_hub_no_disk
from danoliterate.evaluation.execution.model_inference import set_deterministic

def test_model_loading():
    set_deterministic(1887)
    for example_model in "sshleifer/tiny-gpt2", "jonatanklosko/test-tiny-gpt2-sharded":
        # Standard method of loading for reference
        standard_model = AutoModelForCausalLM.from_pretrained(example_model).eval()
        custom_model = from_pretrained_hf_hub_no_disk(example_model).eval()

        assert isinstance(standard_model, type(custom_model)), "Custom model had different type"
        assert standard_model.num_parameters() == custom_model.num_parameters(), "Custom model had different size"
        assert standard_model.config == custom_model.config, "Custom model had different configuration"

        assert standard_model.dtype == custom_model.dtype
        assert standard_model.device == custom_model.device
        assert standard_model.generation_config == custom_model.generation_config

        for name, param in standard_model.named_parameters():
            assert torch.allclose(param, dict(custom_model.named_parameters())[name], atol=1e-7), f"Mismatch in {name}"

        with torch.no_grad():
            input_ids = torch.tensor([[0, 1, 2, 3]])
            standard_output = standard_model(input_ids).logits
            custom_output = custom_model(input_ids).logits

        assert torch.allclose(standard_output, custom_output, atol=1e-7), "Custom model gives different output"
