import spacy, json,re

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
output_dir = "D:\\resume-entities-for-ner\\models"
with open('./resume_data.json', encoding="utf8") as f:
    data = json.load(f)

for index, value in enumerate(data):
    try:
        temp = (value['content'], {"entities": [( val['points'][0]['start'], val['points'][0]['end']+1, val['label'][0]  )  for val in value['annotation']] }  )
    except:
        continue
    TRAIN_DATA.append(temp)

TRAIN_DATA = trim_entity_spans(TRAIN_DATA)

nlp2 = spacy.load(output_dir)
for text, _ in TRAIN_DATA:
    doc = nlp2(text)
    print("Entities", [(ent.text.encode(), ent.label_) for ent in doc.ents])