import os

# os.environ["OPENAI_API_KEY"] = "
# os.environ["OPENAI_BASE_URL"] = ""

import json
import sys
import torch
import itertools
import json
# import clip
import numpy as np
import math
import time
import torch.nn.functional as F
import argparse


from swig_v1_categories import SWIG_INTERACTIONS
from hico_categories import HICO_INTERACTIONS

from collections import OrderedDict
from descriptor_strings import stringtolist
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI
from clip import clip
from generate_prompt import generate_prompt, generate_description_compare, generate_description_overall
from descriptor_strings import wordify, stringtolist


def generate_base_description(args, model, label_to_classname):

    prompt_list =  [generate_prompt(category) for category in label_to_classname]

    for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        while(True):
            try:
                completion =  openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "user", "content": prompt}
                ]
            )
                break
            except:
                time.sleep(3)

        string1 = completion.choices[0].message.content

        key = wordify(label_to_classname[idx])

        string = stringtolist(string1, key)
            
        if len(string) == 0:
                descriptors[key].append(str(key))
        else:
                descriptors[key].append([s for s in string])

    return descriptors

def batch_k_means(data, k, batch_size=1024, max_iters=2):
    """
    Runs the k-means algorithm on the given data in batches using CPU.

    Args:
        data (torch.Tensor): The data to cluster, of shape (N, D).
        k (int): The number of clusters to form.
        batch_size (int): The number of data points in each batch.
        max_iters (int): The maximum number of iterations to run the algorithm for.

    Returns:
        A tuple containing:
        - cluster_centers (torch.Tensor): The centers of the clusters, of shape (k, D).
        - cluster_assignments (torch.Tensor): The cluster assignments for each data point, of shape (N,).
    """
    n = data.shape[0]
    num_batches = (n + batch_size - 1) // batch_size

    # Initialize cluster centers randomly
    np.random.seed(42)
    cluster_centers = data[np.random.choice(n, k, replace=False)]
    cluster_assignments = None

    # Run the algorithm for a fixed number of iterations
    for _ in tqdm(range(max_iters)):
        # Process each batch
        for i in range(0, n, batch_size):
            batch = data[i:i+batch_size]

            # Compute distances between batch and cluster centers using broadcasting
            distances = torch.norm(batch[:, None, :] - cluster_centers[None, :, :], dim=-1)

            # Assign each data point in the batch to the nearest cluster center
            batch_cluster_assignments = torch.argmin(distances, dim=1)

            # Update the cluster centers based on the mean of the assigned points
            for j in range(k):
                mask = batch_cluster_assignments == j
                if mask.any():
                    cluster_centers[j] = batch[mask].mean(dim=0)

    # Compute distances between data and cluster centers using broadcasting
    distances = torch.norm(data[:, None, :].to('cpu') - cluster_centers[None, :, :].to('cpu'), dim=-1)
    # Assign each data point to the nearest cluster center
    cluster_assignments = torch.argmin(distances.cuda(), dim=1)

    return cluster_centers, cluster_assignments

def build_tree_in_loop(class_names, descriptors):
    
    res = descriptors
    num_group, cluster_assignments = devide_to_group(class_names, descriptors)

    label_to_classname_np = np.array(class_names)
    for group_idx in tqdm(range(num_group)):
        tmp_index = torch.where(cluster_assignments == group_idx)[0]  
        categories_group = label_to_classname_np[tmp_index.cpu()]

        if isinstance(categories_group, np.ndarray):
            categories_group = categories_group.tolist()
        if not isinstance(categories_group, list):
            categories_group = [categories_group]

        if len(categories_group) <= args.th and len(categories_group)>=2:
            print("direct comparison")
            print(categories_group)
            res = generate_description_compare(categories_group, res, openai)
        elif len(categories_group) <= 1:
            print("lonely!")
            print(categories_group)
        else:
            print("summary!!")
            res = generate_description_overall(categories_group, res, openai)           
            res = build_tree_in_loop(categories_group, res)


    return res

def devide_to_group(class_names, descriptors):
    
    description_encodings = OrderedDict()
    for k, v in tqdm(descriptors.items()):
        if k in class_names:
            if len(v[-1]) == 0:
                tokens = clip.tokenize('a person is ' + str(k), truncate=True).to(device)
            else:
                tokens = clip.tokenize(v[-1], truncate=True).to(device)
            description_encodings[k] = F.normalize(model.encode_text(tokens))
            
    text_avg_emb = [None]*len(description_encodings)

    for i, (k,v) in enumerate(description_encodings.items()):
        text_avg_emb[i] = v.mean(dim=0)
    try:
        text_avg_emb = torch.stack(text_avg_emb, dim=0)
    except:
        import pdb
        pdb.set_trace()
    
    num_group = int(math.ceil((len(class_names) / args.num_group_div)))
    
    if num_group <= 1:
        num_group=2
    
    print("### start k_means ###")
    _, cluster_assignments = batch_k_means(text_avg_emb, num_group)

    return num_group, cluster_assignments

def generate_tree_embedding(descriptors):

    description_encodings = OrderedDict()
    for k, _ in descriptors.items():
        description_encodings[k] = [] 
    for k, v in descriptors.items():
        for x in v:
            if len(x) == 0:
                tokens = clip.tokenize('a person is ' + str(k), truncate=True).to(device)
            else:
                tokens = clip.tokenize(x, truncate=True).to(device)
            description_encodings[k].append(F.normalize(model.encode_text(tokens)).tolist())


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Build the tree embedding of labels', add_help=False)

    parser.add_argument('--dataset_file', default='swig', choices=['hico', 'swig'])
    parser.add_argument('--clip_model', default="ViT-B/16", type=str, help="Name of pretrained CLIP model")
    parser.add_argument('--num_group_div', default=6, type=int, help="the min mums of group")
    parser.add_argument('--th', default=3, type=int, help="the treshold to compare")
    args = parser.parse_args()

    
    # --variable--
    openai = OpenAI()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    descriptors = defaultdict(list)
    label_to_classname = []

    if args.dataset_file == 'swig':
        for action in SWIG_INTERACTIONS:
            label_to_classname.append(action['name'])
    elif args.dataset_file == 'hico':
        for hoi in HICO_INTERACTIONS:
            hoi_name = " ".join([hoi["action"], hoi["object"]])
            label_to_classname.append(hoi_name.replace("_", ' '))

    model, preprocess = clip.load(args.clip_model, device=device, jit=False, download_root='./')
    model.eval()
    model.requires_grad_(False)
    

    descriptors = generate_base_description(args, model, label_to_classname)
    
    # with open("./tools/swig_base_decriptions.json", "w") as file:
    #    json.dump(descriptors, file)
    # file.close()

    # with open("./tools/swig_base_decriptions.json", "r") as file:
    #    descriptors = json.load(file)
    # file.close()
    

    build_descriptions = build_tree_in_loop(label_to_classname, descriptors)

    # with open("./tools/swig_decriptions.json", "w") as file:
    #    json.dump(descriptors, file)
    # file.close()

    # with open("./tools/swig_decriptions.json", "r") as file:
    #    descriptors = json.load(file)
    # file.close()


    description_encodings = generate_tree_embedding(descriptors)

    description_encodings_final = OrderedDict()
    for k in label_to_classname:
        description_encodings_final[k] = description_encodings[k]

    with open("./tools/" + args.dataset_file + "_decriptions_embedding.json", 'w') as f:
        json.dump(description_encodings_final, f)








    
