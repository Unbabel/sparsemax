import xml.etree.ElementTree as et
import sys
import pdb

filepath_arff = sys.argv[1]
filepath_xml = sys.argv[2]

label_names = set()
tree = et.parse(filepath_xml)
root = tree.getroot()
label_nodes = root.getchildren()
for node in label_nodes:
    label_name = node.attrib['name']
    label_names.add(label_name)

label_fields = dict()
input_fields = dict()
start_data = False
num_attributes = 0
f = open(filepath_arff)
for line in f:
    line = line.rstrip('\r\n')
    if line == '@data':
        print >> sys.stderr, 'Number of input attributes: %d' % (num_attributes - len(label_fields))
        print >> sys.stderr,  'Number of output labels: %d' % (len(label_fields))
        start_data = True
        #pdb.set_trace()
        assert len(label_names) == len(label_fields), pdb.set_trace()
        continue
    elif line.startswith('@attribute'):
        fields = line.split(' ')
        attribute = ' '.join(fields[1:-1])
        attribute = attribute.strip("'")
        attribute = attribute.replace("\\'", "'")
        print >> sys.stderr, attribute
        #attribute = fields[1]
        if attribute in label_names:
            lid = len(label_fields)
            label_fields[num_attributes] = lid
        else:
            fid = len(input_fields)
            input_fields[num_attributes] = fid
        num_attributes += 1
        continue
    elif start_data:
        fields = line.split(',')
        inputs = {}
        labels = set()
        for i, field in enumerate(fields):
            if i in label_fields:
                val = int(field)
                if val == 1:
                    labels.add(str(label_fields[i]))
            else:
                val = float(field)
                inputs[input_fields[i]+1] = val
        if len(labels) == 0:
            print >> sys.stderr, 'Skipping entry with no labels...'
            continue
        print ','.join(labels) + '\t' + ' '.join(['%d:%f' % (key, val) for key, val in inputs.iteritems()])

f.close()
