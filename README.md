和和 (wawa) - Japanese Lyrics Generation
============

## Project Description

和和 is a project to use various generative AIs (LLMs) to produce Japanese song lyrics.

The initial version of the project used only n-grams (NLTK's MLE model) to generate lyrics and a RoBERTa MLM to populate gaps in provided inputs. The n-grams were able to produce some output which improved with higher n, but the time needed to create/load them was becoming too high as n grew.

The intent of 和和 is to use LLMs created for Japanese and tuning them to create song lyrics instead. 
