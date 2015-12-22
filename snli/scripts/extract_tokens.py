import sys
import parse_tree

#gold_label	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	captionID	pairID	label1	label2	label3	label4	label5


filepath = sys.argv[1]
f = open(filepath)
num_skipped = 0
# Skip first line.
f.readline()
for line in f:
    line = line.rstrip('\n')
    fields = line.split('\t')
    label = fields[0]
    if label == '-':
        num_skipped += 1
        print >> sys.stderr, 'Skipping sentence with empty label.'
        continue
    parsed_premise = fields[3]
    parsed_hypothesis = fields[4]
    tree = parse_tree.ParseTree()
    tree.load(parsed_premise)
    words, _ = tree.extract_words()
    premise = ' '.join(words)
    tree = parse_tree.ParseTree()
    tree.load(parsed_hypothesis)
    words, _ = tree.extract_words()
    hypothesis = ' '.join(words)
    print '\t'.join([label, premise, hypothesis])

print >> sys.stderr, 'Skipped %d sentences.' % num_skipped
