import sys
import nltk.translate.bleu_score as bs

from code.data import read_conllu


def read_txt(filename):
    sents = []
    for line in open(filename):
        if line.startswith(u'# text') or line.startswith(u'#text'):
            split = line.split(u'text = ')
            if len(split) > 1:
                text = split[1]
            else:
                #text = '# #'
                text = ''
            sents.append(text.lower().split())
    return sents


def main(gold_file, pred_file, gkey='word', pkey='word'):
    gold_sents = list(read_conllu(gold_file, ud=True, skip_lost=False, simple=True)) # Read UD format, (id, word, lemma, ...) 
    pred_sents = list(read_conllu(pred_file, ud=False, skip_lost=False, simple=True)) # SRST format (id, lemma, word, ...)

    all_ref = [ [[t[gkey].lower() for t in sent.get_tokens()]] for sent in gold_sents]
    all_hyp = [ [t[pkey].lower() for t in sent.get_tokens()] for sent in pred_sents]

    # fallback to text input
    if not all_ref:
        all_ref = [[s] for s in read_txt(gold_file)]
    if not all_hyp:
        all_hyp = read_txt(pred_file)

    # print(len(all_ref), len(all_hyp))

    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(all_ref, all_hyp, smoothing_function=chencherry.method2)
    exact = sum(r[0] == h for r, h in zip(all_ref, all_hyp)) / len(all_ref)
    # print(f'{gkey}: bleu={bleu:.4f}, exact={exact:.4f}')
    print(f'{100*bleu:.2f}')


if __name__ == '__main__':
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Wrong number few arguments.')
        print(str(sys.argv[0]), 'reference-file', 'prediction-file', '[reference-key = WORD/lemma]', '[prediction-key = WORD/lemma]')
        exit()
