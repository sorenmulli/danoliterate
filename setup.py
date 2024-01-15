from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

REQS = {
    "base": "requirements.txt",
    "full": "requirements-full.txt",
}


requires = {}
for version, file in REQS.items():
    with open(file, encoding="utf-8") as f:
        requires[version] = [
            line.strip()
            for line in f.read().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

setup_args = dict(
    name="danoliterate",
    version="0.0.3",
    packages=find_packages(),
    author="Søren Winkel Holm",
    author_email="swholm@protonmail.com",
    install_requires=requires["base"],
    extras_require={
        "full": requires["full"],
    },
    include_package_data=True,
    url = "https://github.com/sorenmulli/danoliterate",
    description="Benchmark of Generative Large Language Models in Danish",
    long_description_content_type="text/markdown",
    long_description=readme,
    license="Apache License 2.0",
)

if __name__ == "__main__":
    setup(**setup_args)
