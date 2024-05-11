#!/bin/bash
source ~/environments/fyp/bin/activate
#this is for getting the moddifiable portions based on the defined rules.
echo "getting modifiable parts"
mkdir output
python3 ./getting_modifiable_parts.py
#get prompts to generate fake named entities
echo "create prompts for generating the fake entities"
python3 create_prompts_for_fake_named.py
#put them into the code for chatgpt thing.
python3 ./create_fake_entities.py
python3 extract_fake_named_entities.py
#generate fake questions based on the entities and other modifiable parts
echo "generating fake questions based on the entities and all"
python3 ./create_real_question_tuples.py
python3 ./create_fake_questions.py
python3 ./create_fake_question_tuples.py
echo "generate fake paragraphs using the above questions"
python3 ./create_fake_para.py other_prompts
python3 ./create_fake_para.py named_prompts
python3 ./fill_in_failed_req.py other
python3 ./fill_in_failed_req.py named
# extract the paragraphs
python3 ./extract_fake_paragraphs.py other
python3 ./extract_fake_paragraphs.py named
#create prompts for llama 2
python3 create_prompts_with_fake_para.py -t other -c 2 --related
python3 create_prompts_with_fake_para.py -t named -c 2 --related
echo "The final prompts for llama-2 are in {DIR}/final_prompts"
echo "The final general prompts are in {DIR}/final_prompts/fake_intermediate"
deactivate
