# Are LLMs Danoliterate?

A benchmark for Generative Large Language Models in Danish. 
To see results and and get more details, check out the leaderboard site:

<p align="center">
<a href="https://danoliterate.compute.dtu.dk/">danoliterate.compute.dtu.dk</a>
</p>

The project is maintained by Søren Vejlgaard Holm at DTU Compute, supported by the Danish Pioneer Centre for AI and with most of the work done as part of the Master's thesis [''Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish''](https://sorenmulli.github.io/thesis/thesis.pdf) supervised by Lars Kai Hansen from DTU Compute and Martin Carsten Nielsen from Alvenir.

## Installation

The package has been developed and used with Python 3.11.
To install the package in a base version, enabling model execution, install
```
pip install danoliterate
```
*Note:* Some features need a full install to run:
```
pip install danoliterate[full]
```

## Usage

See options with
```bash
python -m danoliterate do=evaluate
```

A typical use would be to run your own model hosted on the Huggingface Hub on a scenario, for example the Citizenship Test Scenario (see [the frontend](https://danoliterate.compute.dtu.dk/Scenarios) for scenario descriptions).
Skip the line `scenarios=` to make it run on all scenarios instead.
```bash
python -m danoliterate do=evaluate\
    scenarios="citizenship-test"\
    model.name="MyLittleGPT"\
    model.path="hf-internal-testing/tiny-random-gpt2"\
    evaluation.local_results="./my-result-db"
```

Now, you could share the resulting JSON placed in `my-result-db` to get it included in the Danoliterate benchmark, or you can satisfy your curiosity and score it yourself
```bash
# Calculates scoring metrics
python -m danoliterate do=score\
    evaluation.local_results="./my-result-db"
# Prints them for you
python -m danoliterate do=report\
    evaluation.local_results="./my-result-db"
```

## Contact
Please reach here using GitHub issues or on mail to Søren Vejlgaard Holm either at [swiho@dtu.dk](mailto:swiho@dtu.dk) or [swh@alvenir.ai](mailto:swh@alvenir.ai).
