import json
import data_reader as dr
import pandas as pd


def read_data(_set='train'):
    sentences = []
    with open(dr.data('snli_1.0_{}_filtered.jsonl'.format(_set)), 'r') as f:
        for l in f:
            d = json.loads(l)
            sentences.append({
                'pairID': d['pairID'],
                'sentence2': d['sentence2']
            })
    df_sentence = pd.DataFrame(sentences).set_index('pairID')
    df_label = pd.read_csv(dr.data('snli_1.0_{}_gold_labels.csv'.format(_set))).set_index('pairID')
    return df_label.join(df_sentence)


if __name__ == '__main__':
    df = read_data('train')
    print(df.head())
