#contradiction   A blond-haired doctor and her African american assistant looking threw new medical manuals .    A man is eating pb and j        0.0702376 0.0692678 0.0731519 0.0681309  0.071315 0.0689317  0.068691 0.0685695 0.0683184 0.0683785 0.0679264 0.0685965 0.0685101 0.0999748

import numpy as np
import sys
import pdb

filepath = sys.argv[1]
f = open(filepath)
f_out = open('tmp.html', 'w')

f_out.write('<html>')
for line in f:
    line = line.rstrip('\n')
    fields = line.split('\t')
    assert len(fields) == 4 or len(fields) == 5, pdb.set_trace()
    label = fields[0]
    premise = fields[1]
    hypothesis = fields[2]
    if len(fields) == 5:
        predicted_label = fields[3]
        attention = fields[4]
    else:
        attention = fields[3]
    premise_words = premise.split(' ')
    hypothesis_words = hypothesis.split(' ')
    attention_scores = [float(p) for p in attention.split()]

    assert(len(attention_scores) == len(premise_words)), pdb.set_trace()

    #print label, zip(premise_words, attention_scores), hypothesis
    threshold = 1./len(attention_scores) #1e-8

    f_out.write('<p>[%s|%s] ' % (label, predicted_label))
    for word, score in zip(premise_words, attention_scores):
        if score > threshold:
            f_out.write('<span title="%s"><b>%s</b></span> ' % (str(score), word))
        else:
            f_out.write('<span title="%s">%s</span> ' % (str(score), word))
    f_out.write('[%s]</p>' % hypothesis)

f_out.write('</html>')
f_out.close()    
f.close()
