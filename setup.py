from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    required = [
        line.strip()
        for line in f.read().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup_args = dict(
    name="danoliterate",
    version="0.0.1",
    packages=find_packages(),
    author="Søren Winkel Holm",
    author_email="swholm@protonmail.com",
    install_requires=required,
    description="Benchmark of Generative Large Language Models in Danish",
    long_description_content_type="text/markdown",
    long_description=readme,
    license="All rights reserved",
)

if __name__ == "__main__":
    setup(**setup_args)
