import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import data_reader as dr

import data_exploration


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        """
        :param embedding_dim: 300 (dimension del embedding)
        :param hidden_dim: dimension del hidden
        :param vocab_size: #palabras diferentes (para que?)
        :param tagset_size: #output (3?)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, embeds):
        n = len(embeds)
        lstm_out, _ = self.lstm(embeds.view(n, 1, -1))
        tag_space = self.hidden2tag(lstm_out[-1, ...])
        tag_scores = F.log_softmax(tag_space, dim=1)  # Todo: Creo que hay que sacar dim=1
        return tag_scores  # todo: la loss debería comparar con este


def sampler_evaluator(xs, ys, size):
    def evaluate(model):
        with torch.no_grad():
            idx = np.random.choice(
                np.arange(len(xs)),
                size=size,
                replace=False
            )
            acc = sum(
                y == torch.argmax(model(torch.tensor(x, dtype=torch.float))).item()
                for x, y in zip(np.array(xs)[idx], np.array(ys)[idx])
            )
            return acc/size
    return evaluate


def run():
    xs, ys = data_exploration.get_set('train', data_exploration.load_wiki_vector)
    xs_val, ys_val = data_exploration.get_set('dev', data_exploration.load_wiki_vector)

    EMBEDDING_DIM = 300
    HIDDEN_DIM = 20

    model_lstm = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 3)  # type: LSTMTagger
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model_lstm.parameters(), lr=0.1)

    with torch.no_grad():
        inputs = torch.tensor(xs[0], dtype=torch.float)
        tag_scores = model_lstm(inputs)
        print(tag_scores)

    train_evaluator = sampler_evaluator(xs, ys, 50)
    val_evaluator = sampler_evaluator(xs_val, ys_val, 50)

    print("len train: ", len(xs))
    s = 0
    losses = []
    for epoch in range(3):
        for e, (sentence, tag) in enumerate(zip(xs, ys)):
            model_lstm.zero_grad()
            tag_scores = model_lstm(torch.tensor(sentence, dtype=torch.float))
            loss = loss_function(tag_scores, torch.tensor([tag], dtype=torch.long))
            s += loss.item()
            if (e + 1) % 1000 == 0:
                print('\r' + ' ' * 30 + '\rloss ({}/{}): {:05f}'.format(e + 1, len(xs), s / 1000), end='')
                losses.append(s/1000)
                s = 0

            # if (e + 1) % 50000 == 1000:
            #     tr_ev = train_evaluator(model_lstm)
            #     val_ev = val_evaluator(model_lstm)
            #     print(', train={:05f}, val={:05f}'.format(tr_ev, val_ev))
            loss.backward()
            optimizer.step()

        torch.save(model_lstm.state_dict(), dr.data('model_{}.pt'.format(epoch)))
        with open(dr.data('losses.pickle'.format(epoch)), 'wb') as f:
            pickle.dump(losses, f)

        print('\n')
        predict(model_lstm)
        print(eval_model(xs_val, ys_val)(model_lstm))


def eval_model(xs, ys):
    @torch.no_grad()
    def evaluate(model):
        acc = sum(
            y == torch.argmax(model(torch.tensor(x, dtype=torch.float))).item()
            for x, y in zip(xs, ys)
        )
        return acc / len(xs)

    return evaluate


def predict(model):
    par_id, xs = data_exploration.get_test(data_exploration.load_wiki_vector)

    with open(dr.data('result.csv'), 'w') as f:
        f.write('pairID,gold_label\n')
        for pid, x in zip(par_id, xs):
            res = data_exploration.d_mun_ot[
                torch.argmax(model(torch.tensor(x, dtype=torch.float))).item()
            ]
            f.write('{},{}\n'.format(pid, res))

    print("Done!")


if __name__ == '__main__':
    run()
