import spacy
import re
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
from spacy.pipeline import EntityRuler


nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
patterns = [
    {
        "label": "REL",
        "pattern": [
            {
                "TEXT": {
                    "REGEX": r"(?i)\b(wife|husband|mother|father|son|daughter|aunt|uncle|in-laws|in-law|father in-law|mother in-law|grandfather|grandmother|grandchild|family|partner|ex-partner|elder|maid|bus conductor|close relative|domestic help|couple|doctor|police officer|niece|nephew|neighbour|step-daughter|step-son|teacher|married|step-father|step-mother|friend|student|friends|employee|daughter-in-law)\b"
                }
            }
        ],
    }
]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)


def get_entities(text):
    entities = {}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE" or ent.label_ == "CARDINAL":
            x = re.search(
                "(\d+)\s*(years|year|months|month)", ent.text, re.IGNORECASE
            )
            if x:
                entities[ent.text] = "AGE"
            else:
                entities[ent.text] = ent.label_
        else:
            entities[ent.text] = ent.label_
    return entities


labelled_data = pd.read_csv("./english_1K_sequence_label.csv")
labelled_data.columns
labelled_data["NER"] = labelled_data["summary"].apply(get_entities)
labelled_data1 = labelled_data[labelled_data["lab_final"] == "DV"]
labelled_data1.head()
labelled_data.to_csv("./english_1K_with_NER_tags.csv")

