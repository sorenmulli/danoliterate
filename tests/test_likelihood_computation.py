from argparse import Namespace
import logging
import pytest
import numpy as np
from danoliterate.evaluation.execution.model_inference import set_deterministic
from danoliterate.evaluation.execution.huggingface_inference import HuggingfaceCausalLm

logger = logging.getLogger(__name__)

lm_eval_imported = False
try:
    from lm_eval.models.huggingface import HFLM
    lm_eval_imported = True
except ImportError as error:
    print(error)
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

    reference_implementation = HFLM(TEST_LM_KEY, device="cpu")
    reference_predictions = [score for score, _ in reference_implementation.loglikelihood([Namespace(args=req) for req in requests])]
    assert np.allclose(predictions, reference_predictions, atol=1e-5), f"Mismatch:\n{predictions} vs\n{reference_predictions}"

if __name__ == "__main__":
    test_huggingface_causallm()
