"""
Run this script:

python create_embeddings.py --input *PATH_TO_INPUT_FILE* --out-emb *PATH_TO_SAVE_EMBEDDINGS* --out-rec *PATH_TO_SAVE_EMBEDDINGS_RECORDS*
"""


from transformers import BertTokenizer, BertModel, AutoModel
import torch
import re
import argparse

parser = argparse.ArgumentParser(description='Create the embeddings using HuggingFace BERT data')
parser.add_argument('--input', required=True, help="Path to input file containing text, target and label")
parser.add_argument('--out-emb', required=True, help="Path to save the embeddings data")
parser.add_argument('--out-rec', required=True, help="Path to save records of embeddings")
parser.add_argument('--decoding', action='store_true', default=False,
                    help="Pass this argument if using the records data input files")
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-12_H-768_A-12')
model = AutoModel.from_pretrained("google/bert_uncased_L-12_H-768_A-12", output_hidden_states=True)

DECODE = lambda x: tokenizer.decode(tokenizer.convert_tokens_to_ids(re.sub('_\d+', '', x).split()))
remove_unknown = lambda x: x.replace('[UNK]', '').replace(' [UNK]', '').strip()

def get_embeddings(text):
    global emb_i

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    embeddings = torch.concat(output[2][-5:-1], axis=0).sum(axis=0).detach().numpy()[1:-1]

    return embeddings

with open(args.input) as file:
    data = file.readlines()

decoding = args.decoding

all_embeddings = list()
records = ''

emb_i = 0

for i in range(0, len(data), 3):

    if decoding:
        text = DECODE(data[i].strip())
        target = DECODE(data[i + 1].strip())
    else:
        text = data[i].strip().replace('$T$', '[UNK]')
        target = data[i + 1].strip()
    input_text = text.replace('[UNK]', target)

    record = remove_unknown(text)

    embeddings = get_embeddings(input_text)
    tokens = tokenizer.tokenize(input_text)

    tokens_num = list()
    text_record, target_record = [], []

    text_tokens = tokenizer.tokenize(text)
    target_tokens = tokenizer.tokenize(target)
    for k in range(len(text_tokens)):
        if tokens[k:len(target_tokens) + k] == target_tokens:
            break

    for j in range(len(tokens)):

        if j in range(k, len(target_tokens) + k):
            target_record.append(tokens[j] + f'_{emb_i}')
            if '$T$' not in text_record:
                text_record.append('$T$')
        else:
            text_record.append(tokens[j] + f'_{emb_i}')
        tokens_num.append(tokens[j] + f'_{emb_i}')

        emb_i += 1

    if decoding:
        records += '\n'.join([record, ' '.join(text_record), ' '.join(target_record)]) + '\n' + data[i+2]
    else:
        records += '\n'.join([' '.join(text_record), ' '.join(target_record)]) + '\n' + data[i+2]
    all_embeddings.extend([t + ' ' + ' '.join(e.astype('str')) + '\n' for t, e in zip(tokens_num, embeddings)])

with open(args.out_rec, 'w') as file:
    file.writelines(records.strip())
with open(args.out_emb, 'w') as file:
    all_embeddings[-1] = all_embeddings[-1].strip()
    file.writelines(all_embeddings)

print('Total Created Embeddings:', len(all_embeddings))





