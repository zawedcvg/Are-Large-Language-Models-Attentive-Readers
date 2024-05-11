import json
from tqdm import tqdm
import os
import sys
sys.path.insert(1, '..')
from dotenv import load_dotenv
load_dotenv()
DIR = os.getenv("DIR")


if sys.argv[1] == "other":
    file_to_read = f"{DIR}/fake_other_paragraphs.json"
    file_to_write = f"{DIR}/fake_other_paragraphs_extracted.json"
else:
    file_to_read = f"{DIR}/fake_named_paragraphs.json"
    file_to_write = f"{DIR}/fake_named_paragraphs_extracted.json"


with open(file_to_read, "r") as fp:
    fake_paragraphs = json.load(fp)

answers = []
all_fake_para = []
for i in tqdm(fake_paragraphs):
    # print(i)
    if len(i) == 0:
        print("empty")
        all_fake_para.append([])
        continue
    thing = (i["choices"][0]["message"]["content"])
    if len(thing) == 0:
        print("what")
    index = 1
    is_para = False
    fake_para = []
    fake_answers = []
    for sentence in thing.split("\n"):
        # print(sentence)
        # input()
        if is_para:
            if "fake paragraph" not in sentence.strip().lower():
                continue
            else:
                fake_para.append(sentence[sentence.find(":") + 1:])
            is_para = False

        if "fake answer" in sentence.strip().lower():
            fake_answers.append(sentence[sentence.find(":") + 1:].strip())
            is_para = True
    # print(len(fake_answers))

    # if len(fake_answers) != 2:
        # new_fake = []

        # for sentence in thing.split("\n"):
            # if is_para:
                # if "fake paragraph" not in sentence.strip().lower():
                    # continue
                # else:
                    # fake_para.append(sentence[sentence.find(":") + 1:])
                # is_para = False

            # if "fake answer" in sentence.strip().lower():
                # print(sentence)
                # input()
                # new_fake.append(sentence[sentence.find(":") + 1:])
                # is_para = True
    answers.append(fake_answers)
    if len(fake_para) == 2:
        all_fake_para.append([fake_para, fake_answers])
    else:
        all_fake_para.append([])
        print(input())
print(len(all_fake_para))
print(len(answers))

with open(file_to_write, "w") as fp:
    json.dump(all_fake_para, fp)
