from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
import itertools
import json
from tqdm import tqdm
import os
from random import sample, seed
from collections import defaultdict
seed(42)
from chain_of_thought_prompts import chain_of_thought_prompt_control_subqa_mixtral ,chain_of_thought_prompt_subqa_mixtral, chain_of_thought_prompt_subqa, chain_of_thought_prompt_control_subqa, chain_of_thought_prompt_gpt, chain_of_thought_prompt_control_gpt

from dotenv import load_dotenv
load_dotenv()

DIR = os.getenv("DIR")
final_prompts = os.getenv("FINAL_PROMPTS")


parser.add_argument("-t", "--type", help="type of modifiable part")
parser.add_argument('--related', action=argparse.BooleanOptionalAction)
parser.add_argument("-c", "--count", help="number of paragraphs to be inserted", type=int)
parser.add_argument("-m", "--model", help="the type of model it is being generated for", type=str)
parser.add_argument('--same_type', action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()

DIR = os.getenv("DIR")

type_prompt = args.type
model = args.model
file_all_fake_para = f"{DIR}/fake_{type_prompt}_paragraphs_extracted.json"
file_question_tuples = f"{DIR}/final_intermediate_{type_prompt}.json"

with open(file_question_tuples, "r") as fp:
    question_tuples = json.load(fp)
with open(file_all_fake_para, "r") as fp:
    all_fake_para = json.load(fp)
with open("./data/dev_ori.json", "r") as fp:
    original_question = json.load(fp)
a = (question_tuples[0])

def default_value():
    return []

def create_dictionary_same_index(fake_question_tuples):
    fake_question_dict = defaultdict(default_value)
    #this makes a dictionary of the all the fake question tuple indices
    #which relate to the same main question tuple
    for idx, fake_question_tuple in tqdm(enumerate(fake_question_tuples), total=len(fake_question_tuples)):
        _, _, index_actual_question = fake_question_tuple
        to_insert = all_fake_para[idx]
        if len(to_insert) == 2:
            fake_question_dict[index_actual_question].append(idx)
    return fake_question_dict

def get_required_paragraphs(list_fake_idx, is_related, count):
    req_paragraphs = []
    if is_related:
        all_req_paragraphs = [i for i in itertools.combinations(list_fake_idx, int(count / 2))]
        for combination in all_req_paragraphs:
            intermediate_para = [all_fake_para[idx][0] for idx in combination]
            intermediate_answers = [all_fake_para[idx][1] for idx in combination]
            current_paragraph = [i for i in itertools.chain(*intermediate_para)]
            current_titles = [i for i in itertools.chain(*intermediate_answers)]
            req_paragraphs.append([current_paragraph, current_titles])
    else:
        all_req_permutations = [i for i in (itertools.permutations(list_fake_idx, count))]
        all_req_combinations = [i for i in (itertools.combinations(list_fake_idx, count))]
        thing_to_use = all_req_combinations if (args.same_type and not is_related) else all_req_permutations

        for combination in thing_to_use:
            #using permutations and their indices to ensure that it is a paragraph from the first and then the second
            #mod 2 for the more than 2 case
            if count != 1 and not args.same_type:
                all_req_paragraphs = [all_fake_para[idx][0][pos % 2] for pos, idx in enumerate(combination)]
                intermediate_answers = [all_fake_para[idx][1][pos % 2] for pos, idx in enumerate(combination)]
            else:
                all_req_paragraphs = [all_fake_para[idx][0][1] for _, idx in enumerate(combination)]
                intermediate_answers = [all_fake_para[idx][1][1] for _, idx in enumerate(combination)]

            req_paragraphs.append([all_req_paragraphs, intermediate_answers])
    return req_paragraphs

def is_enough_paragraphs(len_fake_idx, is_related, count):
    if is_related:
        if count % 2 != 0:
            print("For related paragraphs, count should be divisible by 2")
            return False
        return int(count / 2) <= len_fake_idx
    else:
        return count <= len_fake_idx


def creating_datasets_with_fake_paragraphs(fake_question_tuples, is_related, count):
    new_dataset = []
    new_dataset_with_idx = []
    new_dataset_no_changes = []
    for_comparison_all = []

    fake_question_dict = create_dictionary_same_index(fake_question_tuples)

    for index, list_fake_idx in tqdm(fake_question_dict.items(), total=len(fake_question_dict.keys())):
        if not is_enough_paragraphs(len(list_fake_idx), is_related, count): continue
        required_info = get_required_paragraphs(list_fake_idx, is_related, count)
        old_paragraph = original_question[index]
        for current_fake_paragraphs, current_titles in required_info:
            for_comparison = []
            new_dataset_no_changes.append(old_paragraph)
            to_be_replaced_paragraph = deepcopy(old_paragraph)
            titles = [context[0] for context in old_paragraph["context"]]
            req_titles = [context[0] for context in old_paragraph["supporting_facts"]]
            not_req_titles_indices = [index for index, value in enumerate(titles) if value not in req_titles]
            req_titles_indices = [index for index, value in enumerate(titles) if value in req_titles]
            if count > len(not_req_titles_indices): break
            to_replace = sample(not_req_titles_indices, count)
            for index_some, value in enumerate(to_replace):
                to_be_replaced_paragraph["context"][value][1] = [current_fake_paragraphs[index_some]]
                to_be_replaced_paragraph["context"][value][0] = current_titles[index_some]
                for_comparison.append(current_fake_paragraphs[index_some])
            for index_some in req_titles_indices:
                for_comparison.append(old_paragraph["context"][index_some])

            for_comparison_all.append(for_comparison)
            new_dataset.append(to_be_replaced_paragraph)
            new_dataset_with_idx.append([to_be_replaced_paragraph, index])

    return new_dataset, new_dataset_no_changes, for_comparison_all, new_dataset_with_idx

new_dataset, new_dataset_no_changes, for_comparison_all, new_dataset_with_idx = creating_datasets_with_fake_paragraphs(question_tuples, args.related, args.count)

if args.same_type:
    same_type_string = "_same_type"
else:
    same_type_string = ""

isExistGeneralPath = os.path.exists(f"{final_prompts}/hotpotqa_format/")

if not isExistGeneralPath:
    os.makedirs(f"{final_prompts}/hotpotqa_format")

with open(f"./{final_prompts}/hotpotqa_format/fake_intermediate_paragraph_idx_{type_prompt}_{args.related}_{args.count}{same_type_string}.json", "w") as fp:
    json.dump(new_dataset, fp)
with open(f"./{final_prompts}/hotpotqa_format/fake_intermediate_paragraph_idx_no_changes_{type_prompt}_{args.related}_{args.count}{same_type_string}.json", "w") as fp:
    json.dump(new_dataset_no_changes, fp)


if model == "mixtral":
    cot = chain_of_thought_prompt_subqa_mixtral(new_dataset)
    cot_control = chain_of_thought_prompt_control_subqa_mixtral(new_dataset)
    cot_no_change = chain_of_thought_prompt_subqa_mixtral(new_dataset_no_changes)
    cot_control_no_change = chain_of_thought_prompt_control_subqa_mixtral(new_dataset_no_changes)
elif model == "gpt":
    cot = chain_of_thought_prompt_gpt(new_dataset)
    cot_control = chain_of_thought_prompt_control_gpt(new_dataset)
    cot_no_change = chain_of_thought_prompt_gpt(new_dataset_no_changes)
    cot_control_no_change = chain_of_thought_prompt_control_gpt(new_dataset_no_changes)

else:
    cot = chain_of_thought_prompt_subqa(new_dataset)
    cot_control = chain_of_thought_prompt_control_subqa(new_dataset)
    cot_no_change = chain_of_thought_prompt_subqa(new_dataset_no_changes)
    cot_control_no_change = chain_of_thought_prompt_control_subqa(new_dataset_no_changes)

isExistModelPath = os.path.exists(f"{final_prompts}/{model}")
if not isExistModelPath:
    os.makedirs(f"{final_prompts}/{model}")

with open(f"{final_prompts}/{model}/fake_{type_prompt}_cot_{args.related}_{args.count}{same_type_string}.json", "w") as fp:
    json.dump(cot, fp)
with open(f"{final_prompts}/{model}/fake_{type_prompt}_cot_control_{args.related}_{args.count}{same_type_string}.json", "w") as fp:
    json.dump(cot_control, fp)
with open(f"{final_prompts}/{model}/fake_{type_prompt}_cot_no_changes_{args.related}_{args.count}{same_type_string}.json", "w") as fp:
    json.dump(cot_no_change, fp)
with open(f"{final_prompts}/{model}/fake_{type_prompt}_cot_control_no_changes_{args.related}_{args.count}{same_type_string}.json", "w") as fp:
    json.dump(cot_control_no_change, fp)
# with open(f"{DIR}/fake_vs_real_{type_prompt}_{args.related}_{args.count}.json", "w") as fp:
    # json.dump(for_comparison_all, fp)
