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

```
python -m danoliterate do=evaluate
```
