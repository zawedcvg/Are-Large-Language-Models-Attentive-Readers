import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from nltk.corpus import wordnet
import stanza
from tqdm import tqdm
import sys
import os
sys.path.insert(1, '..')

from dotenv import load_dotenv
load_dotenv()

incorrect_indices = []

with open('./actual_indices.txt' , 'r') as f:
    incorrect_indices = f.readlines()


incorrect_indices = [int(i) for i in incorrect_indices]

# nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

DIR = os.getenv("DIR")
from sentence_transformers import SentenceTransformer, util
from create_prompts import create_fake_para_openai
model = SentenceTransformer('all-mpnet-base-v2')
model2 = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')

fill_masker = pipeline("fill-mask", model="roberta-base", top_k=10)
with open(f"{DIR}/other_parts.json", 'r') as fp:
    all_details = json.load(fp)
with open(f"{DIR}/all_fake_named_entities_extracted.json", 'r') as fp:
    all_fake_entities = json.load(fp)
with open(f"{DIR}/subqa_tuples.json", 'r') as fp:
    all_question_tuples = json.load(fp)

limit_sentence_sim = 0.991
limit_word_sim = 0.4
limit_word_possibility = 0.01

#create two different things A and B
# have another piece of code to combine both
#also need to modify the prompts

def get_fake_question_pos(nlp, start_char):
    for i in nlp:
        if i.start_char == start_char:
            return get_wordnet_pos(i.xpos)
    print("start_char does not match for any")
    return -1

def is_too_similar(old_synsets, new_synsets):
    if len(old_synsets) == 0 or len(new_synsets) == 0: return False
    for i in old_synsets:
        for j in new_synsets:
            count = i.path_similarity(j)
            if count > 0.4:
                print(count)
                print(i)
                print(j)



named_questions_with_idx = [[], []]
other_questions_with_idx = [[], []]
all_other_word_sim = []
all_other_sent_sim = []
all_other_score = []
all_incorrect_word_sim = []
all_incorrect_sent_sim = []
all_incorrect_score = []


def get_all_modified_questions(details_modifiable, isFirstPart):
    print(len(all_fake_entities[0]))
    print(len(all_fake_entities[1]))
    if isFirstPart: index_question = 0
    else: index_question = 1
    for index, value in tqdm(enumerate(details_modifiable), total=len(details_modifiable)):
        question_tuple = all_question_tuples[index]
        question = question_tuple[index_question]
        question_embedding = model.encode(question, convert_to_tensor=True)
        question_embedding_2 = model2.encode(question, convert_to_tensor=True)
        # second_question = all_question_tuples[index][1]
        if len(value) != 0:
            for details in value:
                if details[1] == "named_entity":
                    # continue
                    fake_entity_index, start_char, end_char = details[0]
                    word_to_change = question[start_char:end_char]
                    answer_first_hop = question_tuple[2]
                    fake_entity = all_fake_entities[index_question][fake_entity_index]
                    new_question = f"{question[:start_char]}{fake_entity}{question[end_char:]}"
                    isAnswer = (word_to_change ==  answer_first_hop) and not isFirstPart
                    #question_tuple[2] is the answer of the first hop. isAnswer is useful for the pairing of tuples later
                    if isAnswer: continue
                    if not isFirstPart:
                         new_question = new_question.replace(answer_first_hop, "[answer]")

                    named_questions_with_idx[index_question].append([new_question, index])
                else:
                    _, pos, start_char, end_char = details[0]
                    word_to_change = question[start_char:end_char]
                    masked_sentence = f"{question[:start_char]}<mask>{question[end_char:]}"
                    possible_answers = fill_masker(masked_sentence)
                    word_embedding = model.encode(word_to_change, convert_to_tensor=True)
                    answer_first_hop = question_tuple[2]
                    for possible_answer in possible_answers:
                        possible_word_embedding = model.encode(possible_answer["token_str"], convert_to_tensor=True)
                        cosine_scores_words = util.cos_sim(word_embedding, possible_word_embedding)
                        possible_answer_embeddings = model.encode(possible_answer["sequence"], convert_to_tensor=True)
                        possible_answer_embeddings_2 = model2.encode(possible_answer["sequence"], convert_to_tensor=True)
                        cosine_scores_sentences = util.cos_sim(question_embedding, possible_answer_embeddings)
                        cosine_scores_sentences_2 = util.cos_sim(question_embedding_2, possible_answer_embeddings_2)

                        if cosine_scores_words < limit_word_sim and cosine_scores_sentences < limit_sentence_sim and possible_answer["score"] > limit_word_possibility:
                            # print(possible_answer["sequence"])
                            # print(question)
                            isAnswer = (word_to_change ==  question_tuple[2]) and not isFirstPart
                            fake_question = possible_answer["sequence"]
                            if isAnswer: continue
                            if not isFirstPart:
                                fake_question = fake_question.replace(answer_first_hop, "[answer]")
                            other_questions_with_idx[index_question].append([fake_question, index])
                            # print(index)
                            if index in incorrect_indices:
                                print(f"word similarity is {cosine_scores_words}")
                                print(f"sentence similarity is {cosine_scores_sentences}")
                                print(f"sentence similarity 2nd thing is {cosine_scores_sentences_2}")
                                print(possible_answer)
                                print(f"actual question is {question}")
                                all_incorrect_sent_sim.append(cosine_scores_sentences_2.item())
                                # input()
                            else:
                                all_other_sent_sim.append(cosine_scores_sentences_2.item())

                            # local_index += 1
                            break;
    print(np.mean(all_incorrect_sent_sim))
    print(np.mean(all_other_sent_sim))
    print(all_incorrect_sent_sim)
    print(all_other_sent_sim)
    input()
    plt.plot(np.arange(0, len(all_incorrect_sent_sim)), all_incorrect_sent_sim, marker='o', linestyle='-', label='attacked')
    plt.plot(np.arange(len(all_other_sent_sim)), all_other_sent_sim, marker='x', linestyle='--', label='attacked')
    input()
    plt.show()
    # plt.plot(categories1, data2, marker='x', linestyle='--', label='non_attacked')

# print("waiting for an input")
# input()
get_all_modified_questions(all_details[0], True)
get_all_modified_questions(all_details[1], False)

print(len(other_questions_with_idx))
print(len(named_questions_with_idx))
print(other_questions_with_idx[0][0])
print(named_questions_with_idx[0][0])
print(other_questions_with_idx[1][0])
print(named_questions_with_idx[1][0])

# other_fake_para_prompts_first_hop = create_fake_para_openai(other_questions_with_idx[0])
# named_fake_para_prompts_first_hop = create_fake_para_openai(named_questions_with_idx[0])

# other_fake_para_prompts_second_hop = create_fake_para_openai(other_questions_with_idx[1])
# named_fake_para_prompts_second_hop = create_fake_para_openai(named_questions_with_idx[1])

# other_fake_para_prompts = [other_fake_para_prompts_first_hop,
                           # other_fake_para_prompts_second_hop]
# named_fake_para_prompts = [named_fake_para_prompts_first_hop,
                           # named_fake_para_prompts_second_hop]
# print(other_fake_para_prompts[0]["user_prompt"])
# print(other_fake_para_prompts[0]["user_prompt_test"])
# print(other_fake_para_prompts[0]["assistant_prompt_test"])
# print()

with open(f"{DIR}/intermediate_other.json", 'w') as fp:
    json.dump(other_questions_with_idx, fp)
with open(f"{DIR}/intermediate_named.json", 'w') as fp:
    json.dump(named_questions_with_idx, fp)

# with open(f"{DIR}/other_fake_para_prompts.json", 'w') as fp:
    # json.dump(other_fake_para_prompts, fp)
# with open(f"{DIR}/named_fake_para_prompts.json", 'w') as fp:
    # json.dump(named_fake_para_prompts, fp)
