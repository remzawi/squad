import numpy as np
import codecs
import string

BiDAF = 'charembdev.csv'
QANet = 'qanetdev.csv'
ensemble = 'val_submission.csv'

def compute_len(csv, onlyAnswers = False):
    with codecs.open(csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    #print(lines)
    L=0
    nonZeros=0
    n = (len(lines) -1)
    for i in range(1, len(lines)):
        line = lines[i].strip('\n').strip('\r').split(',', maxsplit=1)
        l=sum(word.strip(string.punctuation).isalpha() for word in line[1].split())
        #l=len(line[1].split())
        L += l
        if l > 0 :
            nonZeros+=1
    if onlyAnswers:
        L /= max(nonZeros, 1)
    else:
        L /= max(n, 1)
    return L, nonZeros, n

print('Average length of BiDAF answers including no answers: ', compute_len(BiDAF)[0])
print('Average length of QANet answers including no answers: ', compute_len(QANet)[0])
print('Average length of ensemble answers including no answers: ', compute_len(ensemble)[0])

print('Average length of BiDAF answers excluding no answers: ', compute_len(BiDAF,True)[0])
print('Average length of QANet answers excluding no answers: ', compute_len(QANet,True)[0])
print('Average length of ensemble answers excluding no answers: ', compute_len(ensemble,True)[0])

_, nonzeros, tot = compute_len(BiDAF)
print('Porportion of no answers in BiDAF: ', (tot - nonzeros)/tot)
_, nonzeros, tot = compute_len(QANet)
print('Porportion of no answers in QANet: ', (tot - nonzeros)/tot)
_, nonzeros, tot = compute_len(ensemble)
print('Porportion of no answers in ensemble: ', (tot - nonzeros)/tot)