import loader
import re
from sys import argv


def get_exclamation_marks(data, regex_pattern="!", export=False):
    re_object = re.compile(regex_pattern)
    exclamation_marks = [len(re.findall(regex_pattern, review)) for review in data]
    if export:
        with open('../data/exclamation_mark_count.txt','w') as pc:
            for item in exclamation_marks:
                pc.write(str(item)+'\n')
    else:
        return exclamation_marks

def get_question_marks(data, regex_pattern="\?", export=False):
    re_object = re.compile(regex_pattern)
    question_marks = [len(re.findall(regex_pattern, review)) for review in data]
    if export:
        with open('../data/question_mark_count.txt','w') as qc:
            for item in question_marks:
                qc.write(str(item)+'\n')
    else:
        return question_marks


def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()

    if 'export' in args:
        get_exclamation_marks(data, export=True)
        get_question_marks(data, export=True)

   # print(get_exclamation_marks(data))
   # print(get_exclamation_marks(data))

if __name__ == '__main__':
    main()
