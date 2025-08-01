# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized

from tokenizers import Tokenizer
import json
import argparse
import torch
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=Path, default="../data")
    parser.add_argument("--train_file", type=Path, default="babycosmofine_10M.jsonl")
    parser.add_argument("--valid_file", type=Path, default=None)
    parser.add_argument("--tokenizer_folder", type=Path, default="../tokenizers")
    parser.add_argument("--tokenizer_file", type=Path, default="tokenizer_10M.json")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def tokenize_text(tokenizer, text):
    text = text.strip()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int16)
    return ids


def tokenize_file(input_filename, output_filename, tokenizer):
    tokenized_documents = []
    n_subwords = 0

    for i, line in enumerate(tqdm(input_filename.open('rt'))):
        document = json.loads(line)
        try:
            tokenized_document = tokenize_text(tokenizer, document)
            tokenized_documents.append(tokenized_document)
        except:
            tokenized_document = tokenize_text(tokenizer, document['text'])
            tokenized_documents.append(tokenized_document)            
        n_subwords += len(tokenized_document)

        if i == 0:
            print("Example tokenized document:")
            print(document)
            for token in tokenized_document:
                print(tokenizer.id_to_token(token), end=" ")
            print(flush=True)

    torch.save(tokenized_documents, output_filename)
    print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")


if __name__ == "__main__":
    args = parse_args()

    name = f"_{args.name}" if args.name is not None else ""

    tokenizer_path = args.tokenizer_folder / args.tokenizer_file
    input_train_path = args.data_folder / args.train_file

    # load the tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    output_train_path = input_train_path.with_name(f"{input_train_path.stem}{name}_tokenized.bin")

    tokenize_file(input_train_path, output_train_path, tokenizer)

    if args.valid_file is not None:
        input_valid_path = args.data_folder / args.valid_file
        output_valid_path = input_valid_path.with_name(f"{input_valid_path.stem}{name}_tokenized.bin")
        tokenize_file(input_valid_path, output_valid_path, tokenizer)
