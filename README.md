# Inverted Index

A python 3 project for generating an inverted index for a database documents in the context of an Information Retrieval system.

## Getting started

#### Building the database

First of all, you will need a database, and it need to be organized as follows:

1. A `base.txt` file that contains a list of others file names:
```bash
# base.txt
a.txt
b.txt
c.txt
```

2. N files described in `base.txt` containing the words database, for example:
```bash
# a.txt
Era uma CASA muito
engracada. Não tinha teto,
não tinha nada.
```

```bash
# b.txt
quem casa quer casa.
QUEM não mora em
casa, também quer casa!
```

```bash
# c.txt
quer casar comigo, amor?
quer casar comigo,
faça o favor!
mora na minha casa!
```

Note that these files need to be in the same directory than the `base.txt`.

#### Dependencies

This project only have one dependency, the [NLTK](https://www.nltk.org/). And it can be downloaded as follows:

```bash
$ pip3 install nltk
```

After that, you can download all the content from nltk, or download by specific topics as the code asks.

```python
# Downloading all
>>> import nltk
>>> nltk.download()

# Downloading a specific one
>>> import nltk
>>> nltk.download('<lib name>')
```

#### Executing

Now, you just need to execute the .py file as follows:

```bash
$ python3 inverted-index.py <base.txt file location>
```

After that, the program will generate a file called `indice.txt` containing the processed data.

Done! :)
