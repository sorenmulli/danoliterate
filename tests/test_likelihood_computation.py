import logging
import pytest
import numpy as np
from nlgenda.evaluation.execution.model_inference import HuggingfaceCausalLm, set_deterministic

logger = logging.getLogger(__name__)

lm_eval_imported = False
try:
    from lm_eval.models.huggingface import AutoCausalLM
    lm_eval_imported = True
except ImportError:
    logger.warning("LM Evaluation Harness could not be imported, install with " "`git+https://github.com/EleutherAI/lm-evaluation-harness.git`")

TEST_LM_KEY = "sshleifer/tiny-gpt2"
context = "we are the"
continuations = " champions of the world", " mushrooms of the world", "andlsaaoidjaoidoi"
requests = [(context, cont) for cont in continuations]

@pytest.mark.skipif(not lm_eval_imported, reason="LM Evaluation Harness not imported")
def test_huggingface_causallm():
    set_deterministic()
    inference_to_test = HuggingfaceCausalLm(TEST_LM_KEY, download_no_cache=False, batch_size=2)
    predictions = inference_to_test.likelihoods(requests)

    reference_implementation = AutoCausalLM(TEST_LM_KEY, device="cpu")
    reference_predictions = [score for score, _ in reference_implementation.loglikelihood(requests)]
    print(inference_to_test.pipeline.tokenizer)
    print(reference_implementation.tokenizer)
    assert np.allclose(predictions, reference_predictions, atol=1e-5), f"Mismatch:\n{predictions} vs\n{reference_predictions}"

if __name__ == "__main__":
    test_huggingface_causallm()
