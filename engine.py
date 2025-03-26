# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Suchen for HOI detection
"""
Train and eval functions used in main.py
"""
import math, random
import sys
import torch.nn.functional as F
from typing import Iterable
import torch, torchvision
import utils.misc as utils
from models.model import convert_weights
from datasets import build_evaluator
from utils.visualizer import Visualizer
from fvcore.nn import FlopCountAnalysis, flop_count_table
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    dataset_file: str = "", consider_all_hois: bool = False, description_file_path: str = ""):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    hoi_tree_emb = get_hoi_descriptions(dataset_name=dataset_file, description_file_path=description_file_path)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets, texts, tree_emb = prepare_inputs(images, targets, data_loader, device, hoi_tree_emb)
        if consider_all_hois:
            texts, auxiliary_texts = prepare_text_inputs(model, data_loader.dataset.dataset_texts, device)
        # images.tensors:torch.Size([8, 3, 320, 480]); images.mask: torch.Size([8, 320, 480])
        img_sizes = torch.stack([targets[z]['size'] for z in range(len(targets))], dim=0)
        outputs = model(images.tensors, texts, images.mask, img_sizes, tree_emb) # dict_keys(['logits_per_hoi', 'pred_boxes', 'box_scores', 'attn_maps', 'level_id'])
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, postprocessors, criterion, data_loader, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # Convert applicable model parameters to fp16
    # convert_weights(model)

    # Build evaluator
    evaluator = build_evaluator(args)
    hoi_descriptions_emb = get_hoi_descriptions(dataset_name=args.dataset_file, description_file_path=args.description_file_path)
    # Convert all interaction categories into embeddings, only forward pass once!!
    text_tokens, hoi_descriptions_tree_emb = prepare_text_inputs(model, data_loader.dataset.dataset_texts, device, hoi_descriptions_emb)
    text_features = model.encode_text(text_tokens, pure_words=False)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    if args.use_prompt_hint:
        prompt_hint = model.encode_text(text_tokens, pure_words=True)
        prompt_hint = model.promp_proj(prompt_hint)
    else:
        prompt_hint = torch.zeros(0, model.vision_width).to(text_features.device)
    
    # Inference
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device)
        targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]
        
        bs, c, h, w = images.tensors.shape
        img_sizes = torch.stack([targets[z]['size'] for z in range(len(targets))], dim=0)
        if args.clip_preprocess:
            resized_img = [torchvision.transforms.Resize([224,224])(images.tensors[i][:, :img_sizes[i,0], :img_sizes[i,1]]) for i in range(bs)]
            resized_img = torch.stack(resized_img, dim=0)
            decoder_mask = None
        else:
            resized_img = torchvision.transforms.Resize([224,224])(images.tensors)
            raise NotImplementedError("undefined decoder_mask")
        # vision encoder
        feature_maps, atten = model.encode_image(resized_img, model.multi_scale, model.f_idxs)

        image_0 = model.vision_proj[0](sum(w * s for w, s in zip(model.dgw_weights_s1, feature_maps[0:3])))
        image_1 = model.vision_proj[1](sum(w * s for w, s in zip(model.dgw_weights_s2, feature_maps[3:6])))
        image_2 = model.vision_proj[2](feature_maps[-1])
       
        feature_maps = model.dgw_weights_s3[0] * image_0 + model.dgw_weights_s3[1] * image_1 + model.dgw_weights_s3[2] * image_2
       
        vision_outputs = model.hoi_visual_decoder(feature_maps, mask=decoder_mask, prompt_hint=prompt_hint)
        
        hoi_features = vision_outputs["hoi_features"]
        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        logits_per_hoi = model.logit_scale.exp() * hoi_features @ text_features.t()
        
        image_tree_similarity_cumulative = []
        dot_product_matrix_base_cumulative = []
        image_encodings = F.normalize(vision_outputs["hoi_features"])

        assert text_features.shape[0] == len(hoi_descriptions_tree_emb)
 
        for i, (v) in enumerate(hoi_descriptions_tree_emb):
            dot_product_matrix_base = torch.matmul(image_encodings, torch.tensor(v[0]).T.to(device=image_encodings.device)).mean(dim=-1)
            dot_product_matrix_base_cumulative.append(dot_product_matrix_base)
            if len(v) > 1:
                score = torch.stack([(image_encodings @ torch.tensor(f).T.to(device=image_encodings.device)).mean(dim=-1) for f in v[1:]], dim=-1)
                diffs = score[:, :, 1:] - score[:, :, :-1]
                padded_diffs = F.pad(diffs, (1, 0, 0, 0, 0, 0), value=1)
                mask = padded_diffs > model.eta
                first_false = (mask == False).cumsum(dim=1) >= 1
                mask[first_false] = False
                dot_product_matrix_comp = (score * mask).sum(dim=-1) / mask.sum(dim=-1)
                image_tree_similarity_cumulative.append(dot_product_matrix_comp)
            else:
                image_tree_similarity_cumulative.append(dot_product_matrix_base)

        cumulative_tensor_tree = torch.stack(image_tree_similarity_cumulative, dim=-1)
        cumulative_base_tree = torch.stack(dot_product_matrix_base_cumulative, dim=-1)

        logits_per_hoi = model.lamda * (cumulative_base_tree + logits_per_hoi) / 2 +  (1-model.lamda) * cumulative_tensor_tree

        
        outputs = {
            "logits_per_hoi": logits_per_hoi,
            "pred_boxes": vision_outputs["pred_boxes"],
            "box_scores": vision_outputs["box_scores"],
            "attn_maps": vision_outputs['attn_maps'],
            "visual_atten": atten
        }
        
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # if args.vis_outputs:
        #     visualizer = Visualizer(args)
        #     # visualizer.visualize_preds(resized_img, targets, outputs)
        #     visualizer.visualize_attention(resized_img, targets, outputs)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        results = {int(targets[i]['image_id']): postprocessors(
            {'pred_logits': logits_per_hoi[i], 'pred_boxes': vision_outputs["pred_boxes"][i], 'box_scores': vision_outputs["box_scores"][i]},
            targets[i]['orig_size'],
            data_loader.dataset.text_mapper
        ) for i in range(len(images.tensors))}

        evaluator.update(results)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.save_preds()
    # accumulate predictions from all images
    evaluator.accumulate()
    evaluator.summarize(args.epoch)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.eval_subset:
        from datasets.swig import key_idxs
        import numpy as np
        print("all APs:", evaluator.swig_ap[np.asarray(key_idxs)])
        print("mean AP:", np.mean(evaluator.swig_ap[np.asarray(key_idxs)]))
    return stats, evaluator


def prepare_inputs(images, targets, data_loader, device,  hoi_tree_emb):
    """Prepare model inputs."""
    # image inputs
    images = images.to(device)
    targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

    # text inputs
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    texts = []
    text_inputs = []
    unique_hois = set()
    tree_emb = []

    for t in targets:
        for hoi in t["hois"]:
            # Ensure all texts are unique (no duplicates).
            hoi_id = hoi["hoi_id"]
            if hoi_id in unique_hois:
                continue
            else:
                unique_hois.add(hoi_id)
            action_text, object_text = hoi["text"]
            
            hoi_name = " ".join(hoi["text"])

            tree_emb.append(hoi_tree_emb[hoi_name.replace("_", " ")])

            ## <action, object>
            action_token = _tokenizer.encode(action_text.replace("_", " "))
            object_token = _tokenizer.encode(object_text.replace("_", " "))

            action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
            object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
            texts.append([action_token, object_token])
            # text_inputs.append(action_text + " " + object_text)

    # [specific for HICO-DET], load related hois based on the targets in mini-batch
    if hasattr(data_loader.dataset, 'object_to_related_hois') and hasattr(data_loader.dataset, 'action_to_related_hois'):
        object_to_related_hois = data_loader.dataset.object_to_related_hois
        action_to_related_hois = data_loader.dataset.action_to_related_hois

        related_texts = []
        related_tree_emb = []
        related_text_inputs = []
        unique_actions = set()
        unique_objects = set()
        unique_related_hois = set()
        for t in targets:
            for hoi in t["hois"]:
                hoi_id = hoi["hoi_id"]
                query_action_text, query_object_text = hoi["text"]
                if query_action_text in unique_actions or query_object_text in unique_objects:
                    continue
                else:
                    unique_actions.add(query_action_text)
                    unique_objects.add(query_object_text)

                related_hois = action_to_related_hois[query_action_text]
                for hoi in related_hois:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id in unique_hois:
                        continue
                    if hoi_id in unique_related_hois:
                        continue
                    else:
                        unique_related_hois.add(hoi_id)

                    action_text, object_text = hoi["text"]
                    action_token = _tokenizer.encode(action_text.replace("_", " "))
                    object_token = _tokenizer.encode(object_text.replace("_", " "))
                    action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
                    object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
                    related_texts.append([action_token, object_token])
                    related_text_inputs.append(action_text + " " + object_text)
                    ## hoi descriptions
                    hoi_name = " ".join([action_text, object_text])
                    related_tree_emb.append(hoi_tree_emb[hoi_name.replace("_", " ")])

                related_hois = object_to_related_hois[query_object_text]
                for hoi in related_hois:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id in unique_hois:
                        continue
                    if hoi_id in unique_related_hois:
                        continue
                    else:
                        unique_related_hois.add(hoi_id)

                    action_text, object_text = hoi["text"]
                    action_token = _tokenizer.encode(action_text.replace("_", " "))
                    object_token = _tokenizer.encode(object_text.replace("_", " "))
                    action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
                    object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
                    related_texts.append([action_token, object_token])
                    related_text_inputs.append(action_text + " " + object_text)
                    ## hoi descriptions
                    hoi_name = " ".join([action_text, object_text])
                    tree_emb.append(hoi_tree_emb[hoi_name.replace("_", " ")])

        texts.extend(related_texts)
        tree_emb.extend(related_tree_emb)

    return images, targets, texts, tree_emb


@torch.no_grad()
def prepare_text_inputs(model, texts, device, hoi_tree_emb):
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    text_tokens = []
    tree_emb = []
    for action_text, object_text in texts:
        hoi_name = " ".join([action_text.replace(" ", "_"), object_text])
  
        ## <action, object>
        action_token = _tokenizer.encode(action_text.replace("_", " "))
        object_token = _tokenizer.encode(object_text.replace("_", " "))
        action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
        object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
        text_tokens.append([action_token, object_token])

        tree_emb.append(hoi_tree_emb[hoi_name.replace("_", " ")])

    return text_tokens, tree_emb


def get_flop_stats(model, data_loader):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = _get_model_analysis_input(data_loader)
    flops = FlopCountAnalysis(model, inputs)
    print("Total FLOPs(G)", flops.total() / 1e9)
    print(flop_count_table(flops, max_depth=4, show_param_shapes=False))
    return flops


def _get_model_analysis_input(data_loader):
    for images, targets in data_loader:
        images, targets, texts = prepare_inputs(images, targets, "cuda")
        inputs = (images.tensors, texts, images.mask)
        return inputs


from datasets.swig_v1_categories import SWIG_ACTIONS, SWIG_CATEGORIES, SWIG_INTERACTIONS
from datasets.hico_categories import HICO_INTERACTIONS
import json

def get_hoi_descriptions(dataset_name, description_file_path):
    '''
    return: Dict {hoi_id: List[hoi-description1, ...]}
    '''
    res = {}
    # assert dataset_name in description_file_path

    with open(description_file_path, "r") as f:
        hoi_descriptions = json.load(f)

    # with open(description_file_path, "r") as f:
    #     hoi_descriptions = json.load(f)
    
    # if "swig" in dataset_name:
    #     for hoi in SWIG_INTERACTIONS:
    #         res[hoi["name"]] = hoi_descriptions[hoi["name"]]
    # else:
    #     for hoi in HICO_INTERACTIONS:
    #         hoi_name = " ".join([hoi["action"], hoi["object"]])
    #         res[hoi_name] = hoi_descriptions[hoi_name.replace("_", ' ')]
    return hoi_descriptions
    
''' deprecated, text
def prepare_inputs(images, targets, device):
    """Prepare model inputs."""
    images = images.to(device)
    targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    texts = []
    text_inputs = []
    unique_hois = set()

    for t in targets:
        for hoi in t["hois"]:
            # Ensure all texts are unique (no duplicates).
            hoi_id = hoi["hoi_id"]
            if hoi_id in unique_hois:
                continue
            else:
                unique_hois.add(hoi_id)
            action_text, object_text = hoi["text"]
            action_token = _tokenizer.encode(action_text.replace("_", " "))
            object_token = _tokenizer.encode(object_text.replace("_", " "))

            action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
            object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
            texts.append([action_token, object_token])
            text_inputs.append(action_text + " " + object_text)

    return images, targets, texts
'''