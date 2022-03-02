#!/usr/bin/env bash

perl ~/tools/moses/scripts/tokenizer/tokenizer.perl -a -l de < eval.de.bi.transB4.87 > eval.de.bi.trans1

perl ~/tools/moses/scripts/tokenizer/lowercase.perl