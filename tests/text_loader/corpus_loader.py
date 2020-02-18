import os
import json
import torch
from typing import List


class SentenceSegmenter:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, doc_sentences) -> List[List[str]]:

        token_segments = []
        current_seq = []

        for count, sent in enumerate(doc_sentences):
            if count > 0:
                sent = " " + sent

            token_sent = self.tokenizer.tokenize(sent)

            if len(token_sent) > self.max_seq_length:
                # append last sequence
                token_segments.append(current_seq)

                for i in range(0, len(token_sent) - self.max_seq_length, self.max_seq_length):
                    token_segments.append(token_sent[i:i + self.max_seq_length])

                # assign the current seq the tail of token_sent
                current_seq = token_sent[i + self.max_seq_length:i + self.max_seq_length * 2]
                continue

            if (len(current_seq) + len(token_sent)) > self.max_seq_length:
                token_segments.append(current_seq)
                current_seq = token_sent
            else:
                current_seq = current_seq + token_sent

        if len(current_seq) > 0:
            token_segments.append(current_seq)

        # remove empty segment
        token_segments = [seg for seg in token_segments if seg]

        return token_segments


class CorpusLoader:
    def __init__(self, tokenizer, max_seq_length: int, corpus_path: str, cache_dir: str = ".cache/processed_corpus"):
        self.tokenizer = tokenizer
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.sent_segmenter = SentenceSegmenter(tokenizer, max_seq_length)

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def load_sector(self, sector_id):
        if self.cache_dir:

            cache_path = os.path.join(self.cache_dir, f"{sector_id}_cache.pkl")

            if os.path.exists(cache_path):
                try:
                    processed_docs = torch.load(cache_path)
                    return processed_docs
                except:
                    print("File Corrupted. Data will be re-processed")

        # processing data
        with open(os.path.join(self.corpus_path, str(sector_id) + ".jsonl"), "r") as f:
            data = f.readlines()

        processed_docs = []

        print("Processing Data. Takes about 10 mins")

        for line in data:
            example = json.loads(line)
            token_segments = self.sent_segmenter(example["sents"])

            processed_docs.append(token_segments)

        if self.cache_dir:
            print("Saving Into Cache")
            torch.save(processed_docs, f"{sector_id}_cache.pkl")

        return processed_docs