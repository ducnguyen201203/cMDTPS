from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import math
import torchvision.transforms as T
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("DAPROJECT.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result




class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 grayscale_transform=None, 
                 mim_transform=None, 
                 text_length: int = 77,
                 truncate: bool = True,
                 enhance_mlm:bool=False,
                 prob4newmlm : float=0.15,
                 vision_patch_size:int=16):
        self.dataset = dataset
        self.transform = transform
        self.grayscale_transform =   grayscale_transform
        self.mim_transform       =   mim_transform
        self.text_length         = text_length
        self.truncate            = truncate
        self.enhance_mlm         = enhance_mlm
        self.mask_prob_new_mlm   =  prob4newmlm 
        self.tokenizer = SimpleTokenizer()
        
        self.vision_patch_size = vision_patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        img_ema = read_image(img_path)
        # grayimg = read_image(img_path)
        mimimg = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
            img_ema = self.transform(img_ema)
        # if self.grayscale_transform is not None:
        #     grayimg = self.grayscale_transform(grayimg)
        if self.mim_transform is not None:
            mimimg = self.mim_transform(mimimg)
        # grayscale_caption = self.__remove_color_from_cap(caption)
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # grayscale_caption_tokens = tokenize(grayscale_caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        
        if self.enhance_mlm:
            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels_2(caption_tokens.clone().detach().cpu().numpy())
            # grayscale_mlm_tokens, grayscale_mlm_labels = self._build_random_masked_tokens_and_labels_2(grayscale_caption_tokens.clone().detach().cpu().numpy())
        
        else:
            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.clone().detach().cpu().numpy())
            # grayscale_mlm_tokens, grayscale_mlm_labels = self._build_random_masked_tokens_and_labels(grayscale_caption_tokens.clone().detach().cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            
            # 'gray_images': grayimg,
            # 'gray_caption_ids':grayscale_caption_tokens,
            # 'gray_mlm_ids': grayscale_mlm_tokens,
            # 'gray_mlm_labels': grayscale_mlm_labels,

            'mim_images': mimimg,

            'images_ema': img_ema, 

            "none":True
        }
        return ret

    def __remove_color_from_cap(self, sentence):
        # Extended list of color words
        color_words = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'gray', 'brown', 
               'light brown', 'dark blue', 'pink', 'magenta', 'cyan', 'teal', 'indigo', 'violet', 'beige', 
               'turquoise', 'lavender', 'maroon', 'olive', 'peach', 'salmon', 'tan', 'azure', 'ivory', 
               'teal', 'silver', 'navy blue', 'pea green', 'charcoal', 'aquamarine', 'coral', 'fuchsia', 
               'wheat', 'lime', 'crimson', 'khaki', 'hot pink', 'olden', 'plum', 'olive',  
               'burgundy', 'brick red', 'mahogany', 'vermilion', 'tomato red', 'fire engine red', 'rust', 
               'wine', 'blood red', 'garnet', 'raspberry', 'cranberry', 'scarlet', 'ruby', 'cherry', 
               'brick red', 'rose', 'coral']
        
        # Replace color words with an empty string
        for color in color_words:
            sentence = re.sub(r'\b' + color + r'\b', '', sentence, flags=re.IGNORECASE)

        return " ".join(sentence.split())




    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            label_token = int(token)
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(label_token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
    


    def _build_random_masked_tokens_and_labels_2(self, tokens):
        """
        Masking some (nouns, adjective, verb) tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        selected_categories= ['DT', "NN", "NNS", "JJ", "CD", "PRP"]

        def __is_core_words_(token):
            try:
                if not token in token_range: return False
                word = self.tokenizer.decode([int(token)])
                tokens = nltk.word_tokenize(word)
                post_tag = nltk.pos_tag(tokens)[0][1]
                return post_tag in selected_categories
            except:
                print("mlm2 got error in nltk --> ",word, " \t pos_tag = ", nltk.pos_tag([word]))
                return True     


        labels = []
        for i, token in enumerate(tokens):
            label_token = int(token)
            if 0 < token < 49405 and __is_core_words_(token):
                prob = random.random()
                if prob < self.mask_prob_new_mlm:
                    prob /= self.mask_prob_new_mlm

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(label_token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
    

        ##functions for MIM