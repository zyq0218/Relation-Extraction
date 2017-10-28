'''
Created on 2015-5-7

@author: IRISBEST
'''
import FileUtil
import re
import string
import sys


dir_host = 'C:/Users/IRISBEST/Desktop/CDR/'
dir_server = '/home/BIO/zhaozhehuan/disease_ner/'

Prefix_pattern = '^[A-Z /]+:'



def del_prefixs(text):
    matches = re.findall(Prefix_pattern, text)
    if len(matches) == 1:
        return text.replace(matches[0], '').strip()
    else:
        return text


#replace number and float with num and float
def replace_nums(text):

    num_pattern = '\s\d[,\d]*\s'
    float_pattern = '\s\d+\\.\d+\s'

    replaced_num = 1
    while(replaced_num > 0):
        text, replaced_num = re.subn(num_pattern, ' num ', text)        

    replaced_num = 1
    while(replaced_num > 0):
        text, replaced_num = re.subn(float_pattern, ' float ', text)
    
    return text

    
def process_punctuations(text):
    
    text = text.replace('-',' - ')\
                .replace(', ',' , ')\
                .replace('+',' + ')\
                .replace('>',' > ')\
                .replace('<',' < ')\
                .replace('/ ', '')\
                .replace('/', ' / ')\
                .replace('; ', ' ; ')\
                .replace(': ', ' : ')\
                .replace('? ', ' ? ')\
                .replace("'s ", " 's ")\
                .replace("s' ", "s 's ")\
                .replace("' ", " ")\
                .replace(" '", " ")\
                .replace(")", " ) ")\
                .replace("(", " ( ")\
                .replace("]", " ] ")\
                .replace("[", " [ ")\
                .replace("}", " } ")\
                .replace("{", " { ")\
                .replace('"', '')\
                .replace('!', '')\
                .replace('#', '')\
                .replace('$', '')\
                .replace('&', '')\
                .replace('*', '')\
                .replace('@', '')\
                .replace('~', '')\
                .replace('|', '')\
                .replace('%', '')\
                .replace('  ', ' ')\
                .replace('  ', ' ')\
                .replace('  ', ' ')
    return text



def process_sentence(text):
    
    text = del_prefixs(text)
    
    text = process_punctuations(text)
    
    text = process_parenthses(text)
    
    text = replace_nums(text)
    
    text = text.replace('  ', ' ').strip()
    
    return text.lower()




def process_sentences(sentence_list, doc_id=None):
    new_sent_list = []
    new_entity_list = []

    for sentence in sentence_list:
        processed_sent, entitys = process_sentence(sentence)

        if doc_id is not None:
            new_sent_list.append(processed_sent)
            new_entity_list.append(doc_id + '\t' + entitys)
        else:
            new_sent_list.append(processed_sent)
            new_entity_list.append(entitys)
            
    return new_sent_list, new_entity_list


def is_the_mesures(text):
    
    pat_word = '[a-zA-Z]'
    mat_word = re.findall(pat_word, text)
    if len(mat_word) == 0: # there is no alphabet like 50%; +; -
        return True
        
    pat_num = '\d'
    mat_num = re.findall(pat_num, text)
    if len(mat_num) >= 1:
        if text.lower().find('kg') != -1 or \
            text.lower().find('less than') != -1 or \
            text.lower().find('/day') != -1 or \
            text.lower().find('mg') != -1 or \
            text.lower().find('mug') != -1 or \
            text.lower().find('/l') != -1 or \
            text.lower().find('/ml') != -1 or \
            text.lower().find('/dl') != -1 or \
            text.lower().find('+') != -1 or \
            text.lower().find('+/-') != -1 or \
            text.lower().find('<') != -1 or \
            text.lower().find('=') != -1 or \
            text.lower().find('>') != -1:
            return True
    else:
        return False


# process parentheses of Biocreative V CDR corpus
def process_parenthses(text):
    
    end_index = text.find(')')
    while end_index != -1:
        begin_index = text[:end_index].rfind('(')
        if begin_index != -1:
            if is_the_mesures(text[begin_index+1:end_index]):
                text = text[:begin_index] + text[end_index+1:]
                end_index = text.find(')')
            else:
                end_index = text.find(')', end_index+1)
        else:
            end_index = text.find(')', end_index+1)
    
    return text


def split_2_sents(text):
    
    text = text.replace('          ', ' ')\
                .replace('         ', ' ')\
                .replace('        ', ' ')\
                .replace('       ', ' ')\
                .replace('      ', ' ')\
                .replace('     ', ' ')\
                .replace('    ', ' ')\
                .replace('   ', ' ')\
                .replace('  ', ' ')
                
    sent_list = []
    index = text.find('. ')
    while index != -1:
        pre = text[:index]
        if pre.endswith('i.e') or pre.endswith('i.v') \
            or pre.endswith('vs')\
            or pre.endswith('i.p') or pre.endswith('i.c'):
            index = text.find('. ', index + 1)
        elif text[index+2].strip()[0].isupper():
            sent_list.append(text[:index] + ' .')
            text = text[index+2:]
            index = text.find('. ')
        else:
            index = text.find('. ', index + 1)

    sent_list.append(text.strip()[:-1] + ' .')
    
    return sent_list    
            


def read_medline_corpus(filename):
    
    ascii_only = translator(keep=string.ascii_letters + string.digits + " .,-+<>/;:?'[]{}()")
    abstracts = []
    abstract = []
    for line in open(filename):
        if line == '\n':
            if len(abstract) > 10 and not abstract[0].startswith('Author information'):
                sentences = split_2_sents(' '.join(abstract))
                for sentence in sentences:
                    if len(sentence) > 0 :
                        if len(sentence) < 50 or \
                                sentence.find('Copyright') != -1 or \
                                sentence.find('copyright') != -1 or \
                                sentence.find('http://') != -1 or \
                                sentence.find('Wiley Periodicals') != -1 or \
                                sentence.find('rights reserved') != -1 or \
                                sentence.find('@') != -1:
                            pass
                        else:
                            abstracts.append((ascii_only(process_sentence(sentence))).lower())
            abstract = []
        else:
            abstract.append(line.strip())
    
    return abstracts            



def translator(frm='', to='', delete='', keep=None):
    if len(to) == 1:
        to = to * len(frm)
    
    trans = string.maketrans(frm, to)
    if keep is not None:
        trans_all = string.maketrans('', '')
        delete = trans_all.translate(trans_all, keep.translate(trans_all, delete))
        
    def translate(s):
        return s.translate(trans, delete)
        
    return translate


def filter_non_ascii(input, extra=" .,-+<>/;:?'[]{}()"):

    ascii_only = translator(keep=string.ascii_letters + string.digits + extra)
    
    return ascii_only(input)


if __name__ == '__main__':
    
    dir = 'C:/Users/Think/Desktop/Relations/'
    abstracts = read_medline_corpus(dir + 'pubmed_result.txt')
    FileUtil.writeStrLines(dir + 'abstracts.txt', abstracts)
    
    sent = 'There was also a very significant increase of IP1 after the addition of 1000 mU hCG (p < 0.001) and IP1 and IP3 when 1000 mU hCG plus oxytocin were added (p > 0.001 and p > 0.01, respectively)'
    print sent
    print process_sentence(sent)
