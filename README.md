# Puck [![Build Status](https://travis-ci.org/dlwh/puck.png?branch=master)](https://travis-ci.org/dlwh/puck)

(c) 2014 David Hall

Puck is a high-speed, high-accuracy parser for natural languages.
It's (currently) designed for use with grammars trained with the
Berkeley Parser and on NVIDIA cards.  On recent-ish NVIDIA cards
(e.g. a GTX 680), around 400 sentences a second with a full Berkeley
grammar for length <= 40 sentences.

The current version is 0.1-SNAPSHOT.

Puck is based on the research in two papers:

* David Hall, Taylor Berg-Kirkpatrick, John Canny, and Dan Klein. 2014. Better, Faster, Sparser GPU Parsing. To Appear in Proceedings of the Association for Computational Linguistics.
* John Canny, David Hall, and Dan Klein. 2013. A multi-Teraﬂop Constiuency Parser using GPUs. In
Proceedings of Empirical Methods in Natural Language Processing.


## Documentation

Puck has three main classes. The first is for compiling the GPU representation of a grammar, the second is for parsing with that grammar, and the third is for
experimental use. Running --help with any of these commands will list all options.

### Building Puck

This project can be built with sbt 0.13.  Run `sbt assembly` to create a fat jar in `target/scala-2.10/`

### Compiling a Grammar

The first step in using Puck is to compile a grammar to GPU code. The best way to do this is to run the command

```
sbt "run-main puck.parser.CompileGrammar --textGrammarPrefix textGrammars/wsj_1.gr:textGrammars/wsj_6.gr --grammar grammar.grz"
```

This produces a parser equivalent to the one used in the 2014 paper in a file called `grammar.grz`. The textGrammarPrefix argument accepts
a sequence of plain text grammars, separated by colons. We have provided the cascade of grammars used in the Berkeley Parser for English. In 
practice, using `wsj_1` and `wsj_6` gives you all the benefit for GPU grammars.

### Running the parser

The parser can be run with:

```
sbt "run-main puck.parser.RunParser --grammar grammar.grz <input files>"
```

This will output 1 tree per line. By default it will skip sentences longer than 50 words. 
Input files must be pretokenized into one sentence per line, with words represented as PTB-compatible format. For example:

```
-LRB- Dong-A had had a technology agreement with Jeep maker American Motors Corp. , now a part of Chrysler Corp . -RRB-
```

We plan to fix this soon.

### Experiments

We benchmarked our parser by running it on the treebank.

```bash
sbt "run-main puck.parser.CLParser --maxParseLength 40 --treebank.path /path/to/treebank/wsj --maxLength 40 --numToParse 20000  --reproject false --viterbi true  --cache false --textGrammarPrefix textGrammars/wsj_1.gr:textGrammars/wsj_6.gr --mem 4g --device 680"
```

Should reproduce.


### Acknowledgements

David Hall is supported by a Google PhD Fellowship. Taylor
Berg-Kirkpatrick is supported by a Qualcomm fellowship.  We gratefully
acknowledge the support of NVIDIA Corporation with the donation of
the Tesla K40 GPU used for this research.
