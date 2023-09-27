import torch
import dask
from dask import dataframe as dd
import re
from torch.utils.data import Dataset
import os
import pickle as pk
import numpy as np


def load_parquet_to_dask(file: str) -> dd:
    """
    :file: path to parquet file
    :return: dask dataframe
    """
    data = dask.dataframe.read_parquet(file, delimiter=',')
    return data


def load_csv_to_dask(file: str) -> dd:
    """
    :file: path to parquet file
    :return: dask dataframe
    """
    data = dask.dataframe.read_csv(file, delimiter=',')
    return data
    

def clean_f_terms(f_terms, Stoken='<START F-TERMS>'):
    """
    This function drops all short f-terms which miss the viewpoint or digit and also remove the additional codes.
    It also adds a special token to the beginning of the f-terms
    the cleaned f-terms are returned as a string.
    """
    split_f_terms = f_terms.split(',')
    clean_f_terms = [f_term[:10]+',' for f_term in split_f_terms if len(f_term)>9]
    joined_f_terms = ''.join(clean_f_terms)
    out = Stoken + joined_f_terms
    return out


def clean_df(raw_data_path):
    """
    :raw_data_path: Path to the parquet-file which stores the unprocessed dataset.

    this function loads the raw dataset stored in a parquet file and cleans it from short f-term patents
    Some patents have f-terms which are incompleate (theme, viewpoint or number are missing)
    Patents which contain such a f-term are removed from the dataset
    """
    raw_data = load_parquet_to_dask(raw_data_path)
    # Cleaning the F-terms and adding an start f-term-token
    raw_data['fterms'] = raw_data['fterms'].apply(clean_f_terms, meta=('fterms', 'str'))
    # Combining the abstract and the f-terms
    raw_data['Sample'] = raw_data['appln_abstract'] + raw_data['fterms']
    # dropping unnessesary columns
    clean_data = raw_data[['Sample']]
    return clean_data


def clean_df_agg_with_id(agg_raw_data):
    """
    :raw_data_path: Path to the parquet-file which stores the unprocessed dataset.

    this function loads the raw dataset stored in a parquet file and cleans it from short f-term patents
    Some patents have f-terms which are incompleate (theme, viewpoint or number are missing)
    Patents which contain such a f-term are removed from the dataset
    """
    raw_data = agg_raw_data
    # Cleaning the F-terms and adding an start f-term-token
    raw_data['fterms'] = raw_data['fterms'].apply(clean_f_terms, meta=('fterms', 'str'))
    # Combining the abstract and the f-terms
    raw_data['Sample'] =  '/' + raw_data['appln_abstract'] + raw_data['fterms']
    raw_data['Sample'] = raw_data['appln_fterm_id'].astype(str) + raw_data['Sample'] 
    # dropping unnessesary columns
    clean_data = raw_data[['Sample']]
    return clean_data


def clean_df_agg(agg_raw_data):
    """
    :raw_data_path: Path to the parquet-file which stores the unprocessed dataset.

    this function loads the raw dataset stored in a parquet file and cleans it from short f-term patents
    Some patents have f-terms which are incompleate (theme, viewpoint or number are missing)
    Patents which contain such a f-term are removed from the dataset
    """
    raw_data = agg_raw_data
    # Cleaning the F-terms and adding an start f-term-token
    raw_data['fterms'] = raw_data['fterms'].apply(clean_f_terms, meta=('fterms', 'str'))
    # Combining the abstract and the f-terms
    raw_data['Sample'] = raw_data['appln_abstract'] + raw_data['fterms']
    # dropping unnessesary columns
    clean_data = raw_data[['Sample']]
    return clean_data



class JapPatDataset(Dataset):
    """Dataset containing Japanese patents and their F-Term classification"""
    def __init__(self, data_folder):
        """
        data_file = path to folder containing the text samples
        """
        super(Dataset).__init__()
        self.data_folder = data_folder
        self.l = 1000000        
        
    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        with open(f'{self.data_folder}/{idx}.txt', 'r', encoding='utf-8') as f:
            item = f.read()
        return item


class LabelEmbedding():
    """
    A class to count the occurrence of each individual label.
    It also creates a dict, which contains each label and matches it to a number
    """
    def __init__(self):
        self.dict = {}
        self.r_dict = {}
        self.occurrence = []
        
        
    def __call__(self, label):
        try: 
            emb = self.dict[label]
            self.occurrence[emb] += 1
        except KeyError:
            emb = len(self.dict)
            self.dict[label] = emb
            self.r_dict[emb] = label
            self.occurrence.append(1)
        return emb
    
    def __len__(self):
        return len(self.dict)
    
    def reverse(self, emb):
        return self.r_dict[emb]
        