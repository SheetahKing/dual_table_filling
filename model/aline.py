import torch
from transformers import BertTokenizer, BertModel
import numpy as np
# 该文件是将bert输出的子词进行合并

def check_jing(word):
    list_word = list(word)
    if len(list_word) <= 2:
        return False
    else:
        if list_word[0] == '#' and list_word[1] == '#' and list_word[2] != '#':
            return True
        else:
            return False

def aline_seq(sentences, seq):
    list_sentences = sentences                  
    list_attention_mask = []      
    list_tokens = []    
    list_mask = []     
    list_seq = []       
    subword_num = []    
    path_bert = '../../pretrained_modes/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")    
    for sentence in list_sentences:             
        temp = tokenizer.tokenize(sentence)     
        list_tokens.append(temp)                
    seq_last = []
    for q in seq:
        h = q
        h = h.tolist()
        seq_last.append(h)
    for sent in list_tokens:    
        mask = []  
        for i in range(len(sent) - 1):
            if check_jing(sent[i]) is False and check_jing(sent[i+1]) is True:
                mask.append('#')
            else:
                mask.append(0)
        mask.append(0)     
        mask.insert(0, 0) 
        mask.insert(-1, 0)  
        list_mask.append(mask)
        subword_num.append(mask.count('#'))  
    for i in range(len(subword_num)):           
        single_seq_after = []                   
        single_seq_before = list(seq_last[i])     
        single_mask = list_mask[i]             
        for j in range(len(single_seq_before)):   
            if j < len(single_mask):
                if j == 0:
                    single_seq_after.append(single_seq_before[0])
                else:
                    if single_mask[j] == '#':   
                        new_word_emb = (np.array(single_seq_before[j]) + np.array(single_seq_before[j + 1])) / 2.0  
                        new_word_emb = list(new_word_emb)
                        single_seq_after.append(new_word_emb)
                    elif single_mask[j-1] == "#" and single_mask[j] == 0:   
                        continue
                    elif single_mask[j-1] == 0 and single_mask[j] == 0:   
                        single_seq_after.append(single_seq_before[j])
            else:
                single_seq_after.append(single_seq_before[j])
        if len(single_seq_before) > len(single_seq_after):   
            for q in range(subword_num[i]):
                insert_pos = len(single_seq_after)
                single_seq_after.insert(insert_pos, single_seq_after[insert_pos - 1])
        list_seq.append(single_seq_after)
    maxlen = max_len(list_seq)      
    for c in range(len(subword_num)):   
        single_attention_mask = attention_mask_gen(list_mask[c], maxlen)
        list_attention_mask.append(single_attention_mask)
    seq_tensor = torch.tensor(list_seq, dtype=torch.float32, device='cuda', requires_grad=False)
    attention_mask_tensor = torch.tensor(list_attention_mask, dtype=torch.int64, device='cuda', requires_grad=False)
    return seq_tensor, attention_mask_tensor, maxlen
def attention_mask_gen(single_mask, maxlen):  
    single_attention_mask = []
    jing_count = single_mask.count("#")   
    for i in range(len(single_mask)):
        if single_mask[i] == "#":
            continue
        else:
            single_attention_mask.append(1)
    if (len(single_mask) - jing_count) == maxlen:     
        return single_attention_mask
    elif (len(single_mask) - jing_count) < maxlen:    
        for i in range(len(single_mask) - jing_count, maxlen):
            single_attention_mask.append(0)
    return single_attention_mask
def max_len(list_seq):     
    maxlen = 0
    for i in range(len(list_seq)):
        if maxlen < len(list_seq[i]):
            maxlen = len(list_seq[i])
        else:
            continue
    return maxlen


