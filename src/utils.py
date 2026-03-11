import json
import os
import clip
import numpy as np
import torch
import tqdm


 
if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32
import datasets
 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
             

def save_response_to_json( new_entry, file_path="gpt_response_cirr.json"):
        
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(new_entry)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
 
 
def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



def build_clip(model_name='ViT-B-32' , device="cuda"):
    import clip
    import open_clip
    from torchvision import transforms

    clip_model_name = model_name
    pretraining = {
        'ViT-B-32':'laion2b_s34b_b79k',
        'ViT-B-16':'laion2b_s34b_b88k',
        'ViT-L-14':'laion2b_s32b_b82k',
        'ViT-H-14':'laion2b_s32b_b79k',
        'ViT-g-14':'laion2b_s34b_b88k',
        'ViT-bigG-14':'laion2b_s39b_b160k'
    }

    weight_path = os.path.join(os.getcwd(), "")
    
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretraining[clip_model_name], cache_dir=weight_path)
    clip_model = clip_model.eval().requires_grad_(False).to(device)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    clip_model.tokenizer = tokenizer    
    return clip_model, clip_preprocess


@torch.no_grad()
def text_encoding(device, clip_model, input_captions, batch_size=32, mode='default'):
    print( "text encoding " )

    n_iter = int(np.ceil( len(input_captions)/batch_size))
    predicted_features = []
        
    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
        
        try :
            if hasattr(clip_model, 'tokenizer'):
                tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
            else:
                tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)
        except Exception as e:
            pass 
            print(f"Exception: {e}")

        # input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        #clip_text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        clip_text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)        
        
    return torch.nn.functional.normalize(predicted_features, dim=-1)


from torch.utils.data import Dataset
@torch.no_grad()
def extract_image_features_(device,  dataset, clip_model, flag= 'pool', batch_size = 32,  num_workers= 8):
 
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
    index_features, index_names, target_names, aux_data = [], [], [], []

    target_features = []
    index_rank = None
    for batch in tqdm.tqdm(loader):

        if flag == 'pool':
            images = batch.get('image')
            names = batch.get('image_name')
        else:
            images = batch.get('reference_image')
            names = batch.get('reference_name')   
            target_name = batch.get('target_name')   

            target_image = batch.get('target_image')
            target_image =  target_image.to(device) 
 
        images = images.to(device)
        
        
        with torch.no_grad(),torch.cuda.amp.autocast():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            
            index_names.extend(names)
            
            if flag != 'pool':
                target_ = clip_model.encode_image(target_image)
                target_features.append(target_.cpu())                
                target_names.extend(target_name)

    index_features = torch.vstack(index_features)

    if flag != 'pool':
        target_features = torch.vstack(target_features)

    return index_features, index_names , target_names , target_features

 
@torch.no_grad()
def extract_image_features_pool(device,  dataset, clip_model, batch_size = 32,  num_workers= 8):
 
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)

    index_features, index_names, index_ranks, aux_data = [], [], [], []
 

    # Extract features    
    index_rank = None
    for batch in tqdm.tqdm(loader):
 
        images = batch.get('image')
        names = batch.get('image_name')
 
        images = images.to(device)
        with torch.no_grad(),torch.cuda.amp.autocast():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)
            if index_rank is not None:
                index_ranks.extend(index_rank)
            # if len(aux_data):
            #     aux_data['ref_features'].append(clip_model.encode_image(ref_images.to(device)).cpu())
            #     if hasattr(clip_model, 'tokenizer'):
            #         aux_data['instruct_features'].append(clip_model.encode_text(clip_model.tokenizer(instructions, context_length=77).to(device)).cpu())
            #     else:
            #         aux_data['instruct_features'].append(clip_model.encode_text(clip.tokenize(instructions, context_length=77).to(device)).cpu())
    
    index_features = torch.vstack(index_features)
 
    return index_features, index_names, index_ranks, aux_data

