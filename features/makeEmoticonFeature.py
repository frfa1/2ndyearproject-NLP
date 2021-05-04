import pandas as pd
import loader
from sys import argv

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


def EmoticonSentiment(data, export=False):
    """
    Returns -> emoticon_count: Number of positive emoticons adds 1
    and number of negative emoticons subtracts 1
    """
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

def main():
    args = set(argv)
    data = loader.load_train()['reviewText'].tolist()
    
    if 'export' in args:
        EmoticonSentiment_out = EmoticonSentiment(data, export=True)

if __name__ == "__main__":
    main()