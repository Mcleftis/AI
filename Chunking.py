import nltk
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.tokenize import word_tokenize

sentence="The quick brown fox jumps over the lazy dog"

tokens=word_tokenize(sentence)
tagged=pos_tag(tokens)

grammar="NP:{<DT>?<JJ>*<NN>}"

chunk_parser=RegexpParser(grammar)

tree=chunk_parser.parse(tagged)


tree.draw()
