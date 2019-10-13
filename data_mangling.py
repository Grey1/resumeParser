from __future__ import unicode_literals, print_function
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""


import json
from pprint import pprint

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import re

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data


# training data
TRAIN_DATA = []

with open('./resume_data.json', encoding="utf8") as f:
    data = json.load(f)

for index, value in enumerate(data):
    try:
        temp_dict= {}
        temp_dict["entities"] = []
        for val in value['annotation']:
            if(val['label'][0]!="Name"):
                temp_dict["entities"].append(
                    ( val['points'][0]['start'], val['points'][0]['end']+1, val['label'][0] )
                )
        temp = (value['content'], {"entities":temp_dict["entities"]      }  )
    except:
        continue
    TRAIN_DATA.append(temp)

TRAIN_DATA = trim_entity_spans(TRAIN_DATA)



TRAIN_DATA.extend([
    ("having experience of 9 years in",{"entities":[(21,28, "TExp")]}),
    ("with over 10 years experience in",{"entities":[(10,18, "TExp")]}),
    ("over 10+ years of experience",{"entities":[(5,14, "TExp")]}),
    ("10+ years of experience",{"entities":[(0,9, "TExp")]}),
])


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir='D:\\resume-entities-for-ner\\new-model', n_iter=10):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        # print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        # nlp2 = spacy.load(output_dir)
        # for text, _ in TRAIN_DATA:
        #     doc = nlp2(text)
        #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

# #     # Expected output:
# #     # Entities [('Shaka Khan', 'PERSON')]
# #     # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
# #     # ('Khan', 'PERSON', 1), ('?', '', 2)]
# #     # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
# # # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3)
# #     # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]

