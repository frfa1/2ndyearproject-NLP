import loader
import re
from sys import argv

def get_exclamation_marks(data, regex_pattern="[\w']+|[!?]", export=False):
    
    re_object = re.compile(regex_pattern)
    
    exclamation_marks = [len([word for word in re.findall(regex_pattern, review) if word == '!']) for review in data]

    if export:
        with open('../data/exclamation_mark_count.txt','w') as pc:
            for line in exclamation_marks:
                pc.write(str(line)+'\n')
    
    
    return exclamation_marks

def get_question_marks(data, regex_pattern="[\w']+|[!?]", export=False):
    
    re_object = re.compile(regex_pattern)
    
    question_marks = [len([word for word in re.findall(regex_pattern, review) if word == '?']) for review in data]

    if export:
        with open('../data/question_mark_count.txt','w') as qc:
            for line in question_marks:
                qc.write(str(line)+'\n')
    
    
    return question_marks


def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

    if 'export' in args:
        exclamation_count_out = get_exclamation_marks(data, export=True)
        question_count_out = get_question_marks(data, export=True)

if __name__ == '__main__':
    main()
