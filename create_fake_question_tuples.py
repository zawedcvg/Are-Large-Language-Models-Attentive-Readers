#TODO mix the other and named
import copy
import itertools

import json
import os
from collections import defaultdict

import sys
sys.path.insert(1, '..')
from dotenv import load_dotenv
load_dotenv()
from create_prompts import create_fake_para_openai
from tqdm import tqdm

DIR = os.getenv("DIR")

with open(f"{DIR}/subqa_tuples.json", 'r') as fp:
    all_question_tuples = json.load(fp)
with open(f"{DIR}/intermediate_other.json", 'r') as fp:
    fake_questions_other = json.load(fp)
with open(f"{DIR}/intermediate_named.json", 'r') as fp:
    fake_questions_named = json.load(fp)

def default_value():
    return []


#there can be following mixes.
#Assuming a to be real first subquestion and a' the fake first subquestion
#Assuming b to be real second subquestion and b' the fake second subquestion, We get the following cases
#one fake paragraph based on a'. Next, based on the answer of a', we use b to generate a related fake paragraph.
# if (b') is not a result of modification of the answer from the first hop, we can use (a', b') and (a, b')
#else no point in using b'
#new rules:
# a'b, a'b'
# no ab', as it doesn't work well

def is_in(num, ranges, thing):
    for i in range(num, num+ranges):
        if i in thing:
            return True

    return False

def the_mixer(all_fake_questions):
    index_kinda = 0
    fake_first_hop = all_fake_questions[0]
    fake_second_hop = all_fake_questions[1]
    first_hop_dict = defaultdict(default_value)
    second_hop_dict = defaultdict(default_value)

    for new_question, index in fake_first_hop:
        first_hop_dict[index].append(new_question)

    for new_question, index in fake_second_hop:
        second_hop_dict[index].append(new_question)

    all_fake_tuples = []
    for index, value in tqdm(first_hop_dict.items(), total=len(first_hop_dict)):
        second_hop = copy.deepcopy(second_hop_dict[index])
        second_question_real = all_question_tuples[index][1]
        answer_replaced_second_hop = second_question_real.replace(all_question_tuples[index][2], "[answer]")
        second_hop.append(answer_replaced_second_hop)
        second_hop = filter(lambda x: "[answer]" in x, second_hop)
        new_fake_tuples = [[i[0], i[1], index] for i in itertools.product(value, second_hop)]
        all_fake_tuples = all_fake_tuples + new_fake_tuples
        index_kinda += len(new_fake_tuples)
    return all_fake_tuples

all_fake_tuples_other = the_mixer(fake_questions_other)
all_fake_tuples_named = the_mixer(fake_questions_named)

other_fake_para_prompts = create_fake_para_openai(all_fake_tuples_other)
named_fake_para_prompts = create_fake_para_openai(all_fake_tuples_named)


with open(f"{DIR}/final_intermediate_other.json", 'w') as fp:
    json.dump(all_fake_tuples_other, fp)
with open(f"{DIR}/final_intermediate_named.json", 'w') as fp:
    json.dump(all_fake_tuples_named, fp)

with open(f"{DIR}/other_fake_para_prompts.json", 'w') as fp:
    json.dump(other_fake_para_prompts, fp)
with open(f"{DIR}/named_fake_para_prompts.json", 'w') as fp:
    json.dump(named_fake_para_prompts, fp)
