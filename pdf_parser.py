# from tika import parser


# raw = parser.from_file(pdf_name)
# raw['content'].replace('\n', '')



# # Scipdf

# pip3 install git+https://github.com/titipata/scipdf_parser

# python3 -m spacy download en_core_web_md

import spacy
import textract
from spacy.symbols import ORTH
from os import path, makedirs, listdir

nlp = spacy.load("en_core_web_md")
TOKEN_DIR = "spacy_tokens"
makedirs(TOKEN_DIR, exist_ok=True)

def pdf_to_tokens(pdf):
    text = extract_text(pdf)
    tokens = parse_text(text)
    pdf_name = path.basename(pdf)
    output_file = path.join(TOKEN_DIR, path.splitext(pdf_name)[0] + ".txt")
    write_to_file(tokens, output_file)

def extract_text(pdf):
    text = textract.process(pdf)
    text = text.decode()
    return text

def parse_text(text):
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")

    # Adding exception for et al. tokenization
    special_case = [{ORTH: "al."}]
    nlp.tokenizer.add_special_case("al.", special_case)

    doc = nlp(text)
    
    tokenized_sentences = ""
    for sentence in doc.sents:
        tokenized_sentences += " ".join([token.text for token in sentence]) + "\n"

    return tokenized_sentences

def write_to_file(tokens, filename):
    with open(filename, 'w') as f:
        f.write(tokens)


def tokenize_pdfs(pdf_dir='data/'):
    pdfs = listdir(pdf_dir)
    pdfs = [path.join(pdf_dir, pdf) for pdf in pdfs]
    for pdf in pdfs:
        print(pdf)
        pdf_to_tokens(pdf)
    

if __name__ == '__main__':
    tokenize_pdfs()