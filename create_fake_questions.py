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

DIR = os.getenv("DIR")
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')

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
        if len(value) != 0:
            for details in value:
                if details[1] == "named_entity":
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
                        if possible_answer == None:
                            continue
                        possible_word_embedding = model.encode(possible_answer["token_str"], convert_to_tensor=True)
                        cosine_scores_words = util.cos_sim(word_embedding, possible_word_embedding)
                        possible_answer_embeddings = model.encode(possible_answer["sequence"], convert_to_tensor=True)
                        cosine_scores_sentences = util.cos_sim(question_embedding, possible_answer_embeddings)

                        if cosine_scores_words < limit_word_sim and cosine_scores_sentences < limit_sentence_sim and possible_answer["score"] > limit_word_possibility:
                            isAnswer = (word_to_change ==  question_tuple[2]) and not isFirstPart
                            fake_question = possible_answer["sequence"]
                            if isAnswer: continue
                            if not isFirstPart:
                                fake_question = fake_question.replace(answer_first_hop, "[answer]")
                            other_questions_with_idx[index_question].append([fake_question, index])
                            break;

get_all_modified_questions(all_details[0], True)
get_all_modified_questions(all_details[1], False)


with open(f"{DIR}/intermediate_other.json", 'w') as fp:
    json.dump(other_questions_with_idx, fp)
with open(f"{DIR}/intermediate_named.json", 'w') as fp:
    json.dump(named_questions_with_idx, fp)
