import re
import spacy
import numpy as np

"""
This module contains functions for creating the following features:

1. Elongated - counts presence of words elongated with consequtive similar characters e.g. yeeeees!,nooo
2. Emoticon counts - counts positive and negative emoticons and returns their difference
3. Avg word length - counts average number of characters in the words in a review
4. Irrealis - counts presence of irrealis words (should,could,would)

"""

def create_elongated(docs: list, mode='count', regex_pattern='\w+\'*\w*', export=False) -> list:
    prog = re.compile(regex_pattern)
    regex_elongated = re.compile(r"(.)(\1)+")
    def count_elongated(search_string, re_object, mode):
        elongated_present = [word for word in re_object.findall(search_string.lower()) if regex_elongated.search(word)]
        if mode == 'count':
            return len(elongated_present)
        elif mode == 'binary':
            return 1 if elongated_present else 0  
    n_elongated = [count_elongated(sentence, prog, mode) for sentence in docs]
    if export:
        if mode=='count':
            with open('../data/elongated_count_train.txt','w') as ir:
                for line in n_elongated:
                    ir.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/elongated_bin_train.txt','w') as ir:
                for line in n_elongated:
                    ir.write(str(line)+'\n')
    return n_elongated



def create_emoticon(data, export=False):
    """
    Returns -> emoticon_count: Number of positive emoticons adds 1
    and number of negative emoticons subtracts 1
    """
    # Found on wikipedia (https://en.wikipedia.org/wiki/List_of_emoticons)
    positive_emoticons = {":‑)",":)",":-]",":]",":-3",":3",":->",":>","8-)","8)",":-}",":}",":o)",":c)",":^)",
    "=]","=)",":‑D",":D","8‑D","8D","x‑D","xD","X‑D","XD","=D","=3","B^D",":-))",
    ":'‑)",":')", ":-*",":*",":×", ";‑)",";)","*-)","*)",";‑]",";]",";^)",";>",":‑",";D",":‑P",
    ":P","X‑P","XP","x‑p","xp",":‑p",":p",":‑Þ",":Þ",":‑þ",":þ",":‑b",":b","d:","=p",">:P",
    "|;‑)", "B-)", "#‑)", "<3", "\o/", "^5","o/\o",">_>^ ^<_<", "^_^","(°o°)","(^_^)/","(^O^)／","(^o^)／","(^^)/",
    "(≧∇≦)/","(/◕ヮ◕)/","(^o^)丿","∩(·ω·)∩","(·ω·)","^ω^", ">^_^<","<^!^>","^/^","（*^_^*）","§^.^§","(^<^)",
    "(^.^)","(^ム^)","(^·^)","(^.^)","(^_^.)","(^_^)","(^^)","(^J^)","(*^.^*)","^_^","(#^.^#)","（^—^）",
    "＼(~o~)／","＼(^o^)／","＼(-o-)／"," ヽ(^。^)ノ","ヽ(^o^)丿","(*^0^*)", "(*_*)","(*_*;",
    "(+_+)"," (@_@)","(@_@。","(＠_＠;)","＼(◎o◎)／！", "(*^^)v","(^^)v","(^_^)v","（’-’*)"," (＾ｖ＾)",
    "(＾▽＾)","(・∀・)","(´∀`)","(⌒▽⌒）","(●＾o＾●)","(＾ｖ＾)","(＾ｕ＾)","(＾◇＾)","( ^)o(^ )",
    "(^O^)","(^o^)","(^○^)",")^o^(","(*^▽^*)","(✿◠‿◠)","uwu","UwU", "( ﾟヮﾟ)", "( ﾟдﾟ)", "（ ^_^）o自自o（^_^ ）",
    "ヽ(´ー`)人(´∇｀)人(`Д´)ノ","d(*⌒▽⌒*)b","♪┏(・o･)┛♪┗ ( ･o･) ┓", "^ㅂ^","ヽ(´▽`)/"}

    negative_emoticons = {":‑(",":(",":‑c",":c",":‑<",":<",":‑[",":[",":-||",">:[",":{",":@",":(",";(",
    "D‑':","D:<","D:","D8","D;","D=","DX",":-/",":/",":‑.",">:/","=/",":L","=L",":S",
    ":$","://)","://3", ">:‑)",":)","}:‑)","}:)","3:‑)","3:)",">;)",">:3",";3",
    ":‑###..",":###..", "<:‑|", "',:-|","',:-l", ":E", "<_<",">_>", "</3","<\3",
    "v.v", ">.<", "(>_<)","(>_<)>", "('_')","(/_;)","(T_T)","(;_;)","(;_;","(;_:)","(;O;)","(:_;)","(ToT)","(Ｔ▽Ｔ)",
    ";_;",";-;",";n;",";;","Q.Q,","T.T","TnT","QQ","Q_Q", "(ー_ー)!!","(-.-)","(-_-)","(一一)","(；一_一)",
    "(－‸ლ)","(ーー゛)","(^_^メ)","(-_-メ)","(~_~メ)","(－－〆)","(・へ・)","(｀´)","<`～´>","<`ヘ´>","(ーー;)",
    "(*￣m￣)", "￣|○","STO","OTZ","OTL","orz", "(╯°□°）╯︵ ┻━┻", "(ノಠ益ಠ)ノ彡┻━┻", "︵ヽ(`Д´)ﾉ︵ ┻━┻", 
    "(´･ω･`)", "(´；ω；`)","ヽ(`Д´)ﾉ","(＃ﾟДﾟ)","（ ´,_ゝ`)", "（・Ａ・）", "（ つ Д ｀）", "エェェ(´д｀)ェェエ",
    "＿|￣|○", "(╬ ಠ益ಠ)", "ヽ(ｏ`皿′ｏ)ﾉ"}
    p_count_list = []
    n_count_list = []
    for review in data:
        p_count = 0
        n_count = 0
        tokens = review.split()
        p_count = sum([p_count+1 for i in tokens if i in positive_emoticons])
        n_count = sum([n_count-1 for i in tokens if i in negative_emoticons])
        p_count_list.append(p_count)
        n_count_list.append(n_count)
    emoticon_count = [x + y for x, y in zip(p_count_list, n_count_list)]
    if export:
        with open('../data/emoticon_count.txt','w') as ec:
            for line in emoticon_count:
                ec.write(str(line)+'\n')
    return emoticon_count



def create_irrealis(docs: list, mode='count', regex_pattern='\w+\'*\w*', export=False) -> list:
    irrealis = set(['should','could','would'])
    prog = re.compile(regex_pattern)
    def count_irrealis(search_string, re_object, mode):
        irrealis_present = [word for word in re_object.findall(search_string.lower()) if word in irrealis]
        if mode == 'count':
            return len(irrealis_present)
        elif mode == 'binary':
            return 1 if irrealis_present else 0  
    n_irrealis = [count_irrealis(sentence, prog, mode) for sentence in docs]
    if export:
        if mode=='count':
            with open('../data/irrealis_count_train.txt','w') as ir:
                for line in n_irrealis:
                    ir.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/irrealis_bin_train.txt','w') as ir:
                for line in n_irrealis:
                    ir.write(str(line)+'\n')
    return n_irrealis



def create_review_length(data, regex_pattern="\w+\'*\w*", export=False):
    """
    Returns
        review_length: The length of the review in words (excludes punctuations)
    """
    re_object = re.compile(regex_pattern)
    review_length = [len([word for word in re.findall(regex_pattern, review)]) for review in data]
    if export:
        with open('../data/review_lengths.txt','w') as rl:
            for line in review_length:
                rl.write(str(line)+'\n')
    return review_length



def create_avg_word_length(data, regex_pattern="\w+\'*\w*", export=False):
    re_object = re.compile(regex_pattern)
    #review_length = [sum(len([word for word in re.findall(regex_pattern, review)])) for review in data]
    word_length = [sum(len(word) for word in re.findall(regex_pattern, review)) for review in data] #get length of all words
    review_length = [len([word for word in re.findall(regex_pattern, review)]) for review in data] #get length of review
    avg_word_length = []
    for i, j in zip(word_length, review_length):
        try:
            avg_word_length.append(i/j)
        except ZeroDivisionError:
            avg_word_length.append(0)    
    if export:
        with open('../data/avg_word_lengths.txt','w') as rl:
            for line in avg_word_length:
                rl.write(str(line)+'\n')
    return avg_word_length



def create_negations_present(docs: list, mode='count', regex_pattern='\w+\'*\w*', export=False) -> list:
    with open('../data/negations.txt') as f: # creates a set of negation words 
        negations = set(word.strip() for word in f.readlines())
    prog = re.compile(regex_pattern)
    def count_negations(search_string,re_object,mode):
        negations_present = [word for word in re_object.findall(search_string.lower()) if word in negations]
        if mode=='count':
            return len(negations_present)
        elif mode=='binary':
            return 1 if negations_present else 0  
    n_negations = [count_negations(sentence,prog,mode) for sentence in docs]
    if export:
        if mode=='count':
            with open('../data/negation_count_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/negation_bin_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
    return n_negations



def create_negated_negatives(docs: list, regex_pattern='\w+\'*\w*',mode='count', export=False):
    negations = [word.strip() for word in open('../data/negations.txt').readlines()]
    negative_words = [word.strip() for word in open('../data/negative.txt').readlines()]
    not_negative = set()
    for word in negative_words:
        for negation in negations:
            not_negative.add(' '.join([negation,word]))
    prog = re.compile(regex_pattern)
    def count_negations(search_string,re_object,mode):
        tokens = re_object.findall(search_string)
        pairs = []
        for i in range(len(tokens)-1):
            pairs.append(' '.join([tokens[i],tokens[i+1]]))
        negations_present = [word for word in pairs if word in not_negative]
        if mode=='count':
            return len(negations_present)
        elif mode=='binary':
            return 1 if negations_present else 0  
    n_negations = [count_negations(sentence,prog,mode) for sentence in docs]
    if export:
        if mode=='count':
            with open('../data/negated_negative_count_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/negated_negative_bin_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
    return n_negations




def create_negated_positives(docs: list, regex_pattern='\w+\'*\w*',mode='count', export=False):
    negations = [word.strip() for word in open('../data/negations.txt').readlines()]
    negative_words = [word.strip() for word in open('../data/positive.txt').readlines()]
    not_negative = set()
    for word in negative_words:
        for negation in negations:
            not_negative.add(' '.join([negation,word]))
    prog = re.compile(regex_pattern)
    def count_negations(search_string,re_object,mode):
        tokens = re_object.findall(search_string)
        pairs = []
        for i in range(len(tokens)-1):
            pairs.append(' '.join([tokens[i],tokens[i+1]]))
        negations_present = [word for word in pairs if word in not_negative]
        if mode=='count':
            return len(negations_present)
        elif mode=='binary':
            return 1 if negations_present else 0  
    n_negations = [count_negations(sentence,prog,mode) for sentence in docs]
    if export:
        if mode=='count':
            with open('../data/negated_positive_count_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
        elif mode=='binary':
            with open('../data/negated_positive_bin_train.txt','w') as nf:
                for line in n_negations:
                    nf.write(str(line)+'\n')
    return n_negations



def create_negation_discourse(docs: list, regex_pattern='\w+\'*\w*', export=False) -> list:
    with open('../data/negations.txt') as f: # creates a set of negation words 
        negations = set(word.strip() for word in f.readlines())
    prog = re.compile(regex_pattern)
    def count_negations(search_string,re_object):
        tokens = [word for word in re_object.findall(search_string.lower())]
        n_half = len(tokens) // 2
        first_half,last_half = tokens[:n_half],tokens[n_half:]
        first = 0
        last = 0
        for negation in negations:
            if negation in first_half:
                first = 1
                break
        for negation in negations:
            if negation in last_half:
                last = 1
        return (str(first),str(last))
    positions = [count_negations(sentence,prog) for sentence in docs]
    if export:
        with open('../data/negation_discourse_train.txt','w') as nf:
            for pair in positions:
                nf.write(','.join(pair)+'\n')
    firsts,lasts = [],[]
    for pair in positions:
        firsts.append(pair[0])
        lasts.append(pair[1])
    return firsts,lasts



def create_negation_density(docs: list, regex_pattern=r'\w+\'*\w*', export=False) -> list:
    with open('../data/negations.txt') as f: # creates a set of negation words 
        negations = set(word.strip() for word in f.readlines())
    prog = re.compile(regex_pattern)
    def count_negations(search_string,re_object):
        tokens = re_object.findall(search_string.lower())
        negations_present = [word for word in tokens if word in negations]
        n_tokens = len(tokens)
        if n_tokens:
            return len(negations_present) / n_tokens
        else:
            return 0
    density = [count_negations(sentence,prog) for sentence in docs]
    if export:
        with open('../data/negation_density','w') as nf:
                for line in density:
                    nf.write(str(line)+'\n')
    return density


def create_exclamation_marks(data, regex_pattern="!", export=False):
    re_object = re.compile(regex_pattern)
    exclamation_marks = [len(re.findall(regex_pattern, review)) for review in data]
    if export:
        with open('../data/exclamation_mark_count.txt','w') as pc:
            for item in exclamation_marks:
                pc.write(str(item)+'\n')
    else:
        return exclamation_marks

def create_shoutcase_count(data, export=False):
    """
    Returns -> shoutcase_count: Number of all-caps words of length 3 or more in the review
    """
    shoutcase_count = []
    for review in data:
        # remove words of length 2 or less (Don't want I, A, CD etc. to count as a shoutcase)
        new_review = ' '.join([i for i in review.split() if len(i)>2])

        tokens = new_review.split()
        count = sum(map(str.isupper, tokens))
        shoutcase_count.append(count)

    if export:
        with open('../data/shoutcase_count.txt','w') as scc:
            for line in shoutcase_count:
                scc.write(str(line)+'\n')

    return shoutcase_count



def create_question_marks(data, regex_pattern="\?", export=False):
    re_object = re.compile(regex_pattern)
    question_marks = [len(re.findall(regex_pattern, review)) for review in data]
    if export:
        with open('../data/question_mark_count.txt','w') as qc:
            for item in question_marks:
                qc.write(str(item)+'\n')
    else:
        return question_marks