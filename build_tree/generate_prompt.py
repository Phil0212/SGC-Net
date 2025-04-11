import json
from swig_v1_categories import SWIG_INTERACTIONS

import os
import sys
import torch
import itertools
import json
# import clip
import numpy as np
import math
import time
import torch.nn.functional as F

from collections import OrderedDict
from descriptor_strings import stringtolist
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI
from clip import clip


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing the action of 'a person is {category_name}' in a photo? Only reply the features of the action of '{category_name} and note the output should stratwith '- '. 
A: There are several useful visual features to tell there is the action of 'a person is {category_name}' in a photo(Only reply the features of the action of '{category_name} and note the output should stratwith '- '. ):
-
"""

def wordify(string):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

def generate_prompt_summary(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q:summarize the following categories with one sentence: Salmon, Goldfish, Piranha, Zebra Shark, Whale Shark, Snapper, Swordfish, Bass, Trout?
A:this is a dataset of various fishes

Q:summarize the following categories with one sentence: Smartphone, Laptop, Piranha, Scanner, Refrigerator, Tiger, Bluetooth Speaker, Projector, Printer?
A:this dataset includes different electronic devices

Q:summarize the following categories with one sentence: Scott Oriole, Baird Sparrow, Black-throated Sparrow, Chipping Sparrow, House Sparrow, Grasshopper Sparrow
A:most categories in this dataset are sparrow

Q: summarize the following actions with one sentence: '{category_name}'?
A: 
"""

def generate_prompt_given_overall_feature(category_name: str, over_all: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a Clay Colored Sparrow in a photo in a dataset: This dataset consists of various sparrows?
A: There are several useful visual features to tell there is a Clay Colored Sparrow in a photo:
- a distinct pale crown stripe or central crown patch
- a dark eyeline and a pale stripe above the eye
- brownish-gray upperparts
- conical-shaped bill

Q: What are useful visual features for distinguishing a Zebra Shark in a photo in a dataset: Most categories in this dataset are sharks?
A: There are several useful visual features to tell there is a Zebra Shark in a photo:
- prominent dark vertical stripes or bands
- a sleek and slender body with a long, flattened snout and a distinctive appearance
- a tan or light brown base color on their body
- a long, slender tail with a pattern of dark spots and bands that extend to the tail fin
- dark edges of both dorsal fins

Q: What are useful features for distinguishing the action of '{category_name}' in a photo: {over_all}? Only reply the features of the action of '{category_name} and note the output should stratwith '- '.
A: There are several useful visual features to distinguish there is a action of '{category_name}' in a photo (Only reply the features of the action of '{category_name} and note the output should stratwith '- '.):
-
"""

def generate_description_overall(categories_group, descriptors, openai):
    # string = ', '.join(categories_group)
    string =', '.join(['a person is ' + sub for sub in categories_group])
    prompt = generate_prompt_summary(string)

    while(True):
        try:
            completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
            break
        except:
            time.sleep(3)    

    overall_feature = completion.choices[0].message.content
    
    print("overall_feature", overall_feature)
    print("they are describing", prompt)
    
    prompt_list =  [generate_prompt_given_overall_feature('a person is' + category, overall_feature) for category in categories_group]
    
    for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        while(True):
            try:
                completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "user", "content": prompt}
                ]
            )
                break
            except:
                time.sleep(3)
        
        string = completion.choices[0].message.content

        key =  wordify(categories_group[idx])
        string = stringtolist(string, key)
        
        if len(string) == 0:
            descriptors[key].append(str(key))
        else:
            descriptors[key].append([s for s in string])
    
    return descriptors
      
def generate_prompt_compare(categories_group: str, to_compare: str):
    return f"""Q: What are useful visual features for distinguishing Hooded Oriole from Scott Oriole, Baltimore Oriole in a photo
A: There are several useful visual features to tell there is a Hooded Oriole in a photo:
- distinctive bright orange or yellow and black coloration
- orange or yellow body and underparts
- noticeably curved downwards bill
- a black bib or "hood" that extends up over the head and down the back

Q: What are useful visual features for distinguishing a smartphone from television, laptop, scanner, printer in a photo?
A: There are several useful visual features to tell there is a smartphone in a photo:
- rectangular and much thinner shape
- a touchscreen, lacking the buttons and dials
- manufacturer's logo or name visible on the front or back of the device
- one or more visible camera lenses on the back

Q: What are useful features for distinguishing the action of '{categories_group}' from '{to_compare}' in a photo? Only reply the features of the action of '{categories_group} and note the output should stratwith '- '. 
A: There are several useful visual features for distinguishing the action of '{categories_group}' in a photo (Only reply the features of the action of '{categories_group} and note the output should stratwith '- '):
"""
        
def generate_description_compare(categories_group, descriptors, openai):

    for x in categories_group:

        subtracted_list = ', '.join([ 'a person is ' + y for y in categories_group if y != x])
        prompt = generate_prompt_compare('a person is ' + x, subtracted_list)
    
        while(True):
            try:
                completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "user", "content": prompt}
                ]
            )
                break
            except:
                time.sleep(3)

        key = wordify(x)
        res = completion.choices[0].message.content
        res = stringtolist(res, key)

        if len(res) == 0:
            descriptors[key].append(str(key))
        else:
            descriptors[key].append([s for s in res])

    return descriptors
