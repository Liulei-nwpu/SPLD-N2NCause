import re
import readline


def get_relation(label):
    relations = re.findall('cause-effect\((.*)\)', label)[0]
    print(relations)
    entitys = relations.split(',')
    i = 0
    while i < len(entitys):
        e1 = entitys[i].lstrip('(')
        e2 = entitys[i + 1].rstrip(')')
        i += 2
        print(e1,e2)

def get_entitys(sentence):
    arguements = {}
    tag = []
    for i in range(1,26):
        e = 'e' + str(i)
        e_1 = '<{}>'.format(e)
        e_2e = '</{}>'.format(e)
        pattern = '<{}>(.*)</{}>'.format(e, e)
        e_all = re.findall(pattern,sentence)
        if len(e_all) > 0:
            arguements[e] = e_all[0]
            tag.append(e)
        else:
            continue
        # need to pay attention to
        sentence.replace(e_1,"").replace(e_2e,"")
    # return self.arguements
    return sentence


fr = open('train.txt','r',encoding='utf-8')
line = fr.readline()
cnt = 0
while line:
    ls = line.split('\t')
    cnt += 1
    print(ls)
    label = ls[2].strip('\n')
    if label != 'noncause':
        
        get_relation(label)
        get_entitys(ls[1])
    print(cnt)
    line = fr.readline()




