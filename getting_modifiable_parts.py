import stanza
import os
from dotenv import load_dotenv
import sys
sys.path.insert(1, '..')
import json
from tqdm import tqdm
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,mwt,pos,lemma,depparse')

with open("./data/dev_sub1.json", "r") as fp:
    single_hop_questions = json.load(fp)
with open("./data/dev_sub2.json", "r") as fp:
    second_hop_questions = json.load(fp)

def check_if_in_between(word, list_to_check):
    for index, i in enumerate(list_to_check):
        if word.start_char >= i.start_char and word.end_char <= i.end_char:
            return ([index, i.start_char, i.end_char], True)
    return (None, False)

def is_related_appos(doc, obj):
    for i in doc.sentences[0].dependencies:
        if (i[0].id == obj.id or i[2].id == obj.id) and i[1] == "appos":
            return True

    return False

def getting_main_obj(doc, dependecies, words, wh_sent):
    if(words[wh_sent[0] - 1].head == 0):
        for first, dep, second in doc.sentences[0].dependencies:
            if first.id == wh_sent[0] and (dep == "nsubj" or dep == "nsubj:pass"):
                return second

    verb_things = ["VBD", "VB", "VBG", "VBN", "VBP", "VBZ"]
    noun_things = ["NN", "NNS", "NNP", "NNPS"]

    for first, dep, second in dependecies:
        if second.id == wh_sent[0] and (dep == "det" or dep == "nsubj" or dep == "nsubj:pass"):
            if first.xpos in noun_things:
                return first
            else:
                thing_to_check = first
                for first, dep, second in doc.sentences[0].dependencies:
                    if thing_to_check.id == second.id and dep == "acl:relcl":
                        return first



        elif second.id == wh_sent[0]:
            pros_obj = first
            if pros_obj.xpos in noun_things:
                return first
            elif pros_obj.xpos in verb_things:
                for first, dep, second in doc.sentences[0].dependencies:
                    if pros_obj == first:
                        if dep == "nsubj" or dep == "nsubj:pass":
                            return second

    return None

def get_modifiable_part(doc, obj, main_obj):
    answer = []
    to_compare_id = None if main_obj is None else main_obj.id
    for i in doc.sentences[0].dependencies:
        if i[0].id == obj.id:
            if i[1] in ["nummod", "amod", "nmod", "compound", "flat"] and (not is_related_appos(doc, i[2])) and to_compare_id != i[2].id:
                answer.append([i[2], i[1]])
    return answer


def pretty_print_deps(sent_dict):
    print ("{:<15} | {:<10} | {:<15} ".format('Token', 'Relation', 'Head'))
    print ("-" * 50)
    for word in sent_dict:
      print ("{:<15} | {:<10} | {:<15} "
             .format(str(word['text']),str(word['deprel']), str(sent_dict[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')))

def get_modifiable_parts(doc):
    words = doc.sentences[0].words
    seen = set()

    wh_things = ["WDT", "WP", "WP$", "WRB"]
    wh_sent = []
    count = 0
    other_things = []
    for i in words:
        if i.xpos in wh_things:
            wh_sent.append(i.id)
            count += 1

    if count != 1:
        return doc.ents, other_things
    else:
        # if wh word is the root, just get the nsubj
        dependencies = doc.sentences[0].dependencies
        main_obj = getting_main_obj(doc, dependencies, words, wh_sent)
        if main_obj is not None:
            print(f"The main object is {main_obj.text}")
        flag = 0
        objs = []
        for first, dep, second in dependencies:
            # if the dependency is an obj/obl, change the things that modify the obj/obl
            if (dep == "obj" or dep == "obl" or dep == "nsubj" or dep == "nsubj:pass") and (second != main_obj):
                has_appos = is_related_appos(doc, second) or is_related_appos(doc, first)
                if has_appos:
                    continue
                answer = get_modifiable_part(doc, second, main_obj)
                objs.append([second, answer])
                flag = 2

        if flag != -1:
            objs_num = list(filter(lambda x: len(x[1]) != 0, objs))
            if len(objs_num) != 0:
                for i in objs_num:
                    for j in i[1]:
                        ner, is_named = check_if_in_between(j[0], doc.ents)
                        if is_named:
                            if [ner, "named_entity"] not in other_things:
                                other_things.append([ner, "named_entity"])
                        else:
                            if j[0] not in seen:
                                seen.add(j[0])
                                #i is the thing being modified and j is the thing modifying it which will be changed later to create the fake questions
                                other_things.append([[j[0].text, j[0].xpos, j[0].start_char, j[0].end_char], "other_part"])

            return doc.ents, other_things

        else:
            return doc.ents, other_things

def get_named_other_tuple(to_loop):
    all_named = []
    all_other = []
    all_things = []
    complete_index_named = 0
    for i in tqdm(to_loop):
        doc = nlp(i["question"])
        named_entities, other_parts = get_modifiable_parts(doc)
        named_entities = [[i.text, i.start_char, i.end_char, i.type] for i in named_entities]
        all_named.append(named_entities)
        for other_part in other_parts:
            if other_part[1] == "named_entity":
                other_part[0][0] += complete_index_named
        all_other.append(other_parts)
        all_things.append([all_named, all_other])
        # input()
        complete_index_named += len(doc.ents)
    return [all_named, all_other]


if __name__ == "__main__":

    load_dotenv()
    DIR = os.getenv('DIR')
    single_hop_info = get_named_other_tuple(single_hop_questions)
    second_hop_info = get_named_other_tuple(second_hop_questions)

    complete_info_named = [single_hop_info[0], second_hop_info[0]]
    complete_info_other = [single_hop_info[1], second_hop_info[1]]


    json_content = {}


    if len(sys.argv) < 2:
        other_parts = "other_parts.json"
        all_named_entities = "all_named_entities.json"

    else:
        other_parts = sys.argv[1]
        all_named_entities = sys.argv[2]

    with open(f"{DIR}/{other_parts}", 'w') as fp:
        json.dump(complete_info_other, fp)

    with open(f"{DIR}/{all_named_entities}", 'w') as fp:
        json.dump(complete_info_named, fp)
