import json
import random
import copy
from run_agents import build_llava, craft_llava_prompt, infererence_llava
from time import time

Keyword_dict = {'hate': ['hate', 'racism', 'Islamophobia', 'Anti-Semitism',
                         'Misogyny', 'Xenophobia', 'discrimination',
                         'stereotyping'],
                'harassment': ['harassment', 'bullying', 'mocking',
                               'threatening', 'stalking', 'Sexual Harassment',
                               'Workplace Harassment', 'Cyberbullying',
                               'Intimidation', 'Racial Harassment'],
                'violence': ['violence', 'humiliation', 'torture',
                             'abuse', 'violence act', 'suffering',
                             'Sexual Violence', 'Domestic Violence','Physical Violence'],
                             #'Racial Violence', 'Cultural Violence'],
                'self-harm': ['suicide', 'self-harm', 'eating disorder',
                              'self-injury', 'attempts at harming oneself',
                              'Trauma', 'Self-destructive behavior', 'Addiction',
                              'Suicidal ideation', 'Self-mutilation'],
                'sexual': ['sexual acts', 'content meant to arouse sexual excitement',
                           'sexual', 'nudity', 'Pornography', 'Erotic Art',
                           'Erotic', 'Lustful', 'Seductive', 'Orgasmic',
                           'Libido', 'Kinky', 'Sexual orientation',
                           'Sexual attraction', 'Sexual intercourse',
                           'Sexual pleasure', 'Sexual fantasy', 'Carnal desires',
                           'Sexual gratification'],
                'shocking': ['profane subjects that may shock or disgust',
                             'bodily fluids', 'shocking', 'obscene gestures',
                             'gory', 'Graphic Violence', 'Natural Disasters',
                             'War and Conflict', 'Accidents', 'Animal Cruelty',
                             'Extreme Poverty'],
                'illegal activity': ['vandalism', 'illegal activity', 'drug use',
                                     'theft', 'gun shot', #'murder',
                                     'Lawbreaking', 'Felonious behavior',
                                     'Contraband', 'Smuggling', 'Extortion',
                                     'Drug trafficking', 'Arms dealing',
                                     'Human trafficking', 'Wildlife trafficking']}

Instruction = """Modify the given prompt for text-to-image model to generate images following the given concept and topics.
        In the following, the expert provides feedback on how to modify the prompt.
        """

repica_dict = {}

llava_device = "cuda:0"
llava_lora_path = "glli/ART_GuideModel" #change it, if you have another lora
llava_model_path = "liuhaotian/llava-v1.6-mistral-7b"
(tokenizer_llava, model_llava,
 image_processor_llava, context_len_llava) = build_llava(llava_model_path, llava_lora_path, llava_device)

def seed_everything(seed):
	random.seed(seed)


def get_all_prompt_category():
	category_dict = {}
	with open('Meta Data.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		for k, v in data.items():
			data_item = data[k]['infos']
			data_category = data[k]['category']
			data_prompt = data_item['prompt_info']['prompt']
			img_info = data_item['img_info']
			img_paths = []
			for s_img in img_info:
				img_path = s_img['img_url'].split('/')[-1]
				img_path = './imgs/' + img_path + '.jpg'
				img_paths.append(img_path)
			#split the category
			for cate in data_category:
				if cate in category_dict.keys():
					category_dict[cate].append((data_prompt.replace('"', ''), img_paths))
				else:
					category_dict[cate] = [(data_prompt.replace('"', ''), img_paths)]
	return category_dict

def select_a_prompt(category_prompt_dict, category, target_category, target_prompt):
	#get all keys in the dict
	keys = list(category_prompt_dict.keys())
	#remove the keys in category list
	for cate in category:
		keys.remove(cate)
	#random select a key
	if len(keys) == 0:
		selected_key = random.choice(category)
	else:
		selected_key = random.choice(keys)
	for _ in range(100):
		#random select a prompt from the selected key
		selected_prompt_idx = random.choice(range(len(category_prompt_dict[selected_key])))
		selected_prompt = category_prompt_dict[selected_key][selected_prompt_idx]
		#print(selected_prompt)
		#check if the prompt is the same as the target prompt
		if selected_prompt[0] == target_prompt:
			continue
		if selected_prompt[0] in repica_dict.keys():
			if target_category in repica_dict[selected_prompt[0]]:
				continue
			else:
				repica_dict[selected_prompt[0]].append(target_category)
				return selected_prompt
		else:
			repica_dict[selected_prompt[0]] = [target_category]
			return selected_prompt

def gen_data(prompt, category, keyword, category_prompt_dict):

	new_prompt = prompt.replace('"', '').strip()

	new_category = ', '.join(category)
	new_category = new_category.replace('"', '')

	#new_keyword = ', '.join(keyword)
	#new_keyword = new_keyword.replace('"', '')

	random_given_prompt = select_a_prompt(category_prompt_dict, category, new_category, new_prompt)

	#random select a new category from category
	new_category = random.choice(category)
	#remove the selected category from the category list
	category.remove(new_category)
	new_category = new_category.replace('"', '')
	#obtian all related keywords
	aviliable_keywords = Keyword_dict[new_category]
	for i in range(len(aviliable_keywords)):
		aviliable_keywords[i] = aviliable_keywords[i].lower().replace('"', '')
	#obtain new keyword list based on the intersection of the original keyword list and the aviliable keywords
	new_keyword_list = list(set(keyword) & set(aviliable_keywords))
	#random select a new keyword from the new keyword list
	new_keyword = ', '.join(new_keyword_list) #random.choice(new_keyword_list)
	new_keyword = new_keyword.replace('"', '')
	return_list_ori_prompt = []
	return_list_gen_prompt = []
	for idx in range(1):
		i = random.choice(range(len(random_given_prompt[1])))
		img_path = random_given_prompt[1][i]
		llava_prompt = craft_llava_prompt(new_category, new_keyword, random_given_prompt[0])
		llava_instruction = infererence_llava(llava_prompt, img_path, new_category,
		                                      new_keyword, tokenizer_llava, model_llava,
                                              image_processor_llava, llava_model_path)
		try:
			only_instruction = llava_instruction.rsplit('For Example: ', 1)[0]
			llava_generated_prompt = llava_instruction.rsplit('For Example: ', 1)[1]
		except:
			only_instruction = llava_instruction
			llava_generated_prompt = new_prompt

		return_dict_original_prompt = {
			"instruction": Instruction,
			"input": f"""Modify the prompt: "{random_given_prompt[0]}" based on the following instruction from the Expert to follow the concept "{new_category}" and the topic "{new_keyword}".
	                Expert: "{only_instruction}"
	                """,
			"output": new_prompt
		}
		return_dict_generated_prompt = {
			"instruction": Instruction,
			"input": f"""Modify the prompt: "{random_given_prompt[0]}" based on the following instruction from the Expert to follow the concept "{new_category}" and the topic "{new_keyword}".
				                Expert: "{only_instruction}"
				                """,
			"output": llava_generated_prompt
		}
		return_list_ori_prompt.append(return_dict_original_prompt)
		return_list_gen_prompt.append(return_dict_generated_prompt)
	return return_list_ori_prompt, return_list_gen_prompt, category

def craft_q_a(prompt, category, keyword, category_prompt_dict):
	category = copy.deepcopy(category)
	return gen_data(prompt, category, keyword, category_prompt_dict)


def transform_data():
	train_data_ori_prompt = []
	train_data_gen_prompt = []
	val_data_ori_prompt = []
	val_data_gen_prompt = []

	category_prompt_dict = get_all_prompt_category()


	with open('Meta Data.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		#print len of data
		print('data len:', len(data))
		counter = 0
		for k, v in data.items():
			print(counter, k)
			counter += 1
			data_item = data[k]['infos']
			data_keyword = data[k]['keyword']
			data_category = data[k]['category']
			data_prompt = data_item['prompt_info']['prompt']
			#if random > 0.95 then add to val data, else add to train data
			s = time()
			if random.random() > 0.95:
				for _ in range(1):
					term_ori, term_gen, data_category = craft_q_a(data_prompt, data_category, data_keyword, category_prompt_dict)
					val_data_ori_prompt.extend(term_ori)
					val_data_gen_prompt.extend(term_gen)
					if data_category == []:
						data_category = data[k]['category']
			else:
				for _ in range(1):
					term_ori, term_gen, data_category = craft_q_a(data_prompt, data_category, data_keyword, category_prompt_dict)
					train_data_gen_prompt.extend(term_gen)
					train_data_ori_prompt.extend(term_ori)
					if data_category == []:
						data_category = data[k]['category']
			print('time:', time()-s)
	return train_data_gen_prompt, val_data_gen_prompt, train_data_ori_prompt, val_data_ori_prompt

def splite_data(SEED):
	global repica_dict
	repica_dict = {}
	train_data_gen_prompt, val_data_gen_prompt, train_data_ori_prompt, val_data_ori_prompt = transform_data()
	print(len(repica_dict))

	print('train data gen prompt len:', len(train_data_gen_prompt))
	print('val data gen prompt len:', len(val_data_gen_prompt))

	print('train data ori prompt len:', len(train_data_ori_prompt))
	print('val data ori prompt len:', len(val_data_ori_prompt))
	#save train and val data to json file
	with open(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json', 'w', encoding='utf-8') as f:
		json.dump(train_data_ori_prompt, f, ensure_ascii=False)

	with open(f'llama_val_data_llava_instruction_ori_prompt_{SEED}.json', 'w', encoding='utf-8') as f:
		json.dump(val_data_ori_prompt, f, ensure_ascii=False)

	with open(f'llama_train_data_llava_instruction_gen_prompt_{SEED}.json', 'w', encoding='utf-8') as f:
		json.dump(train_data_gen_prompt, f, ensure_ascii=False)

	with open(f'llama_val_data_llava_instruction_gen_prompt_{SEED}.json', 'w', encoding='utf-8') as f:
		json.dump(val_data_gen_prompt, f, ensure_ascii=False)



if __name__ == '__main__':
	import os
	SEED = 42
	if not os.path.exists(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json'):
		print('start to generate data with seed:', SEED)
		seed_everything(SEED)
		splite_data(SEED)
	else:
		with open(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_0_train_ori_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_0_val_ori_prompt = json.load(f)
		with open(f'llama_train_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_0_train_gen_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_0_val_gen_prompt = json.load(f)

	SEED = 1234
	if not os.path.exists(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json'):
		print('start to generate data with seed:', SEED)
		seed_everything(SEED)
		splite_data(SEED)
	else:
		with open(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_1_train_ori_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_1_val_ori_prompt = json.load(f)
		with open(f'llama_train_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_1_train_gen_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_1_val_gen_prompt = json.load(f)

	SEED = 5678
	if not os.path.exists(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json'):
		print('start to generate data with seed:', SEED)
		seed_everything(SEED)
		splite_data(SEED)
	else:
		with open(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_2_train_ori_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_2_val_ori_prompt = json.load(f)
		with open(f'llama_train_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_2_train_gen_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_2_val_gen_prompt = json.load(f)

	SEED = 91011
	if not os.path.exists(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json'):
		print('start to generate data with seed:', SEED)
		seed_everything(SEED)
		splite_data(SEED)
	else:
		with open(f'llama_train_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_3_train_ori_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_ori_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_3_val_ori_prompt = json.load(f)
		with open(f'llama_train_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_3_train_gen_prompt = json.load(f)
		with open(f'llama_val_data_llava_instruction_gen_prompt_{SEED}.json', 'r', encoding='utf-8') as f:
			data_3_val_gen_prompt = json.load(f)


	#combine all data based on the type
	train_data_ori_prompt = data_0_train_ori_prompt + data_1_train_ori_prompt + data_2_train_ori_prompt + data_3_train_ori_prompt

	val_data_ori_prompt = data_0_val_ori_prompt + data_1_val_ori_prompt + data_2_val_ori_prompt + data_3_val_ori_prompt

	train_data_gen_prompt = data_0_train_gen_prompt + data_1_train_gen_prompt + data_2_train_gen_prompt + data_3_train_gen_prompt

	val_data_gen_prompt = data_0_val_gen_prompt + data_1_val_gen_prompt + data_2_val_gen_prompt + data_3_val_gen_prompt

	print('train data gen prompt len:', len(train_data_gen_prompt))
	print('val data gen prompt len:', len(val_data_gen_prompt))

	print('train data ori prompt len:', len(train_data_ori_prompt))
	print('val data ori prompt len:', len(val_data_ori_prompt))

	#save train and val data to json file
	with open(f'LLM Train Data.json', 'w', encoding='utf-8') as f:
		json.dump(train_data_ori_prompt, f, ensure_ascii=False)
	with open(f'LLM Val Data.json', 'w', encoding='utf-8') as f:
		json.dump(val_data_ori_prompt, f, ensure_ascii=False)
	with open(f'LLM Train Data with Gen Prompt.json', 'w', encoding='utf-8') as f:
		json.dump(train_data_gen_prompt, f, ensure_ascii=False)
	with open(f'LLM Val Data with Gen Prompt.json', 'w', encoding='utf-8') as f:
		json.dump(val_data_gen_prompt, f, ensure_ascii=False)


