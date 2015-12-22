from span import *
import re
import pdb

class ParseTreeConstituent(Span):
    def __init__(self, start, end, name='', tree=None):
        Span.__init__(self, start, end, name)
        self.tree = tree
        
        
class ParseTree:
    def __init__(self):
        self.label = ''
        self.children = list() # This is a list of ParseTree objects.
        
    def is_leaf(self):
        return len(self.children) == 0
        
    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_leaf()
        
    def get_label(self):
        return self.label
        
    def get_children(self):
        return self.children

    def add_child(self, tree):
        self.children.append(tree)

    def get_post_order_traversal(self, extract_null_words=False):
        if not extract_null_words and self.get_label() == '-NONE-':
            return []
        trees = []
        for kid in self.children:
            trees.extend(kid.get_post_order_traversal(extract_null_words))
        trees.append(self)
        return trees

    def strip_decorations(self):
        for child in self.get_children():
            child.strip_decorations()
        if not self.is_leaf() and not self.is_preterminal():
            self.label = re.sub('[-|=].*', '', self.get_label())

    def collapse_singleton_spines(self, same_label_only=True,
                                  append_labels=False):
        # Merge vertical spines with the same label into a single node.
        # E.g. "(NP (NP ... ))" becomes (NP ...).
        for child in self.get_children():
            child.collapse_singleton_spines(same_label_only, append_labels)
        if len(self.children) == 1 and not self.is_preterminal():
            if same_label_only:
                if self.label == self.children[0].label:
                    self.children = self.children[0].children
            else:
                if self.label != '' and append_labels:
                    #pdb.set_trace()
                    self.label += '|' + self.children[0].label
                else:
                    self.label = self.children[0].label
                self.children = self.children[0].children

    def add_constituents_recursively(self, constituents, index, extract_null_words=False):
        if self.is_preterminal():
            if extract_null_words or self.get_label() != '-NONE-':
                child =  self.get_children()[0]
                constituent = ParseTreeConstituent(index, index, child.get_label(), child)
                constituents[child] = constituent
                constituent = ParseTreeConstituent(index, index, self.get_label(), self)
                constituents[self] = constituent
                return 1 # Length of leaf constituent.
            else:
                return 0
        else:
            next_index = index
            for kid in self.get_children():
                next_index += kid.add_constituents_recursively(constituents, next_index, extract_null_words)
            constituent = ParseTreeConstituent(index, next_index-1, self.get_label(), self)
            constituents[self] = constituent
            return next_index - index

    def get_constituents(self, extract_null_words=False):
        constituents = dict()
        self.add_constituents_recursively(constituents, 0, extract_null_words)
        return constituents

    def __str__(self):
        desc = ''
        if not self.is_leaf():
            desc += '('
        desc += self.get_label()
        if not self.is_leaf():
            for i, child in enumerate(self.get_children()):
                if i > 0 or self.get_label() != '':
                    desc += ' '
                desc += child.__str__()
            desc += ')'
        return desc

    def extract_words(self, extract_null_words=False):
        words = []
        tags = []
        if self.is_preterminal():
            if extract_null_words or self.get_label() != '-NONE-':
                tags.append(self.get_label())
                words.append(self.get_children()[0].get_label())
        else:
            for i, child in enumerate(self.get_children()):
                child_words, child_tags = child.extract_words(extract_null_words)
                words.extend(child_words)
                tags.extend(child_tags)
        return words, tags

    def eliminate_null_words(self):
        new_children = []
        for kid in self.children:
            if kid.is_leaf():
                new_children.append(kid)
            elif kid.get_label() != '-NONE-':
                kid.eliminate_null_words()
                if not kid.is_leaf():
                    new_children.append(kid)
        self.children = new_children

    def load(self, desc):
        left_bracket = '('
        right_bracket = ')'
        whitespace = ' '
        name = ''
        tree_stack = []
        line = re.sub('[\n\r\t\f]', ' ', desc)
        line = re.sub('\)', ') ', line)        
        line = re.sub('\(', ' (', line)
        while True:
            line_orig = line
            line = re.sub('\( \(', '((', line)
            line = re.sub('\) \)', '))', line)
            if line == line_orig:
                break
        line = re.sub('[ ]+', ' ', line)
        line = line.strip(' ')
#        print line
#        pdb.set_trace()
        for j, ch in enumerate(line):
            if ch == left_bracket:
                name = ''
                if len(tree_stack) > 0:
                    tree = ParseTree()
                    tree_stack[-1].add_child(tree)
                else:
                    tree = self
                tree_stack.append(tree)
            elif ch == right_bracket:
                tree = tree_stack.pop()
            elif ch == whitespace:
                pass
            else:
                name += ch
                assert j+1 < len(line) and line[j+1] != left_bracket, pdb.set_trace()
                if line[j+1] == right_bracket:
                    # Finished a terminal node.
                    tree = ParseTree()
                    tree.label = name
                    tree_stack[-1].add_child(tree)
                elif line[j+1] == whitespace:
                    # Just started a non-terminal or a pre-terminal node.
                    tree_stack[-1].label = name
                    name = ''
            
        assert len(tree_stack) == 0, pdb.set_trace()
        #assert line == str(self), pdb.set_trace()

