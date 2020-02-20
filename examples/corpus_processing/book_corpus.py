"""
Preprocessing the Book Corpus
"""


import os
import sys
import glob
import torch
import spacy
import ray
import json
import tqdm
import warnings
from typing import List
from utils import merge_short_sents



ray.init(num_cpus=64)


DATA_PATH = "../RAW/books_txt/"
TEMP_PATH = "book_temp"
STORE_PATH = ""


dir_paths = os.listdir(DATA_PATH)


all_filenames = []

for dir_path in dir_paths:
    dir_path = os.path.join(DATA_PATH, dir_path)
    # we only care about txt files
    filenames = glob.glob(os.path.join(dir_path, "*.txt"))
    all_filenames.extend(filenames)


print(f"total number: {len(all_filenames)}")


# initialize SpaCy
spacy.prefer_gpu()
# we only need sentencizer
nlp = spacy.load("en", disable=["ner", "tagger", "parser", 
                                "merge_noun_chunks", "merge_entities", "merge_subtokens",
                               "entity_ruler", "textcat", "entity_linker"])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp.max_length = 1212504981


filename = all_filenames[0]


with open(filename, "r") as f_read:
    data = f_read.read()
    
spacy_doc = nlp(data)
doc_sents = [sent.text for sent in spacy_doc.sents]
doc_sents = merge_short_sents(doc_sents)


print("Processed Example")
print(doc_sents)


def process_document(filename:str) -> List[str]:
    with open(filename, encoding="utf8", errors='ignore') as f_read:
        data = f_read.read()

    spacy_doc = nlp(data)
    doc_sents = [sent.text for sent in spacy_doc.sents]
    doc_sents = merge_short_sents(doc_sents)
    
    return doc_sents
        
@ray.remote
def write_data(filenames:List[str], file_descripts:List[str]=None, output_name=None, sector_id=None):
    ""
    f_write = open(output_name, "w+")
    # write file descriptions

    
    if sector_id == 0:
        pbar = tqdm.tqdm(filenames)
    else:
        pbar = filenames
    
    for i, filename in enumerate(pbar):

        desp = file_descripts[i][0]
        
        try:
            doc_sents = process_document(filename)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
            
        # dump into json
        line = {"id": desp, 
                "sents": doc_sents}
        line = json.dumps(line)
        
        # write into jsonl
        f_write.write(line)
        f_write.write("\n")
    
    f_write.close()
    
    return 0


# define the sector size
total_num_sector = 64


sector_path = os.path.join(STORE_PATH, "SECTOR")
meta_path = os.path.join(STORE_PATH, "META")

# make directories
os.makedirs(sector_path, exist_ok=True)
os.makedirs(meta_path, exist_ok=True)

file_descripts = []

# write descriptions
for sector_id in range(total_num_sector):
    sector_filenames = all_filenames[sector_id::total_num_sector]
    file_ids = ["book_corpus_" + str(i + sector_id * total_num_sector) 
                for i in range(sector_id, len(all_filenames), total_num_sector)]
    file_descripts.extend([(file_id, os.path.join(sector_path, str(sector_id) + ".jsonl"))
                            for file_id in file_ids])

assert len(all_filenames) == len(file_descripts) == len(set([item[0] for item in file_descripts]))


# Fire Ray Remote
remote_objs = []

for sector_id in range(total_num_sector):
    sector_filenames = all_filenames[sector_id::total_num_sector]
    
    sector_file_ids = ["book_corpus_" + str(i + sector_id * total_num_sector) 
                for i in range(sector_id, len(all_filenames), total_num_sector)]
    
    sector_file_descripts = [(file_id, os.path.join(sector_path, str(sector_id) + ".jsonl"))
                            for file_id in sector_file_ids]
    
    obj_id = write_data.remote(sector_filenames, 
                               sector_file_descripts, 
                               os.path.join(sector_path, f"{sector_id}.jsonl"), 
                               sector_id)
    remote_objs.append(obj_id)
        
for sector_id in range(total_num_sector):
    assert ray.get(remote_objs[sector_id]) == 0






