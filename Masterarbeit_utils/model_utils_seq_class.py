import torch
import psutil
import os
import pickle as pk
from transformers import AutoTokenizer, OPTForCausalLM, OPTForSequenceClassification
from tokenizers.processors import TemplateProcessing


def get_tokenizer(dump_dir):
    
    # Loading the dict containing all unique f-terms in the datase
    if os.path.isfile(f'{dump_dir}/label_embedding_no_single.pk'):
        with open(f'{dump_dir}/label_embedding_no_single.pk', 'rb') as f:
            dataset_emb = pk.load(f)
            f_term_dict = dataset_emb.dict
    else:
        with open(f'{dump_dir}/f_terms_in_ds_dir.pk', 'rb') as f:
            f_term_dict = pk.load(f)
        
    # Loading a dict, which contains all uniqe f-terms with crawled definitions
    with open(f'{dump_dir}/f_term_dict.pk', 'rb') as f:
        definitions = pk.load(f)
        
    # Loading the original tokenizer for the galactica model
    tokenizer = load_pretrained_Tokenizer('mini')
    
    # Checking for which f-term form the dataset a f-term definition is present
    exceptions = {}
    exceptions_l = 0
    for i, key in enumerate(f_term_dict.keys()):
        try: 
            _ = definitions[key]
            exceptions[key] = 0
        except KeyError:
            exceptions[key] = 1
            exceptions_l += 1
    tokenizer.add_tokens(['<START F-TERMS>', '<ENDĠF-TERMS>'])
    #tokenizer.add_tokens(['<START F-TERMS>'])
    unique_tokens = [key +',' for key, value in exceptions.items() if value ==0] 
    tokenizer.add_tokens(unique_tokens)
    # Adding the start_sequence, end_sequence and padding tokens to the tokenizer
    tokenizer.pad_token = '<pad>'
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    tokenizer._tokenizer.post_processor = TemplateProcessing(
    	single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
    	special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
	)
    # Changing the padding side, because the last token must allways be no pad token 
    tokenizer.padding_side = 'left'
    return tokenizer


def load_pretrained_Tokenizer(model_name):
    """
    :param model_name:  Name of the matching pretrained model
    :return:            Tokenizer matching to the pretrained model
    """

    # A dict to map the correct model urls
    HF_MAPPING = {
        "mini": ("facebook/galactica-125m", torch.float32),
        "base": ("facebook/galactica-1.3b", torch.float32),
        "standard": ("facebook/galactica-6.7b", torch.float32),
        "large": ("facebook/galactica-30b", torch.float32),
        "huge": ("facebook/galactica-120b", torch.float16)}

    return AutoTokenizer.from_pretrained(HF_MAPPING[model_name][0])


def extract_embedding(model):
    """
    :param model:  Loaded Pretrained model
    :return:       Token embeddings
    """
    return model.get_input_embeddings()


def create_embedding(original_embedding: torch.nn.Embedding, n_f_terms: int, device: str) -> torch.nn.Embedding:
    """
    This function takes the original_embedding instance of an OPT model,
        (nn.Embedding instance).
    and the number of f-terms it should embedd (n_f_terms) and creates a new embedding which has
    new weights for all f_terms stacked ontop of the old weigths used for the original tokens

    returns: torch.nn.Embedding
    """
    # calculating parameters for the new embedding instance
    embedding_dim = original_embedding.embedding_dim
    num_embeddings = original_embedding.num_embeddings + n_f_terms
    padding_idx = original_embedding.padding_idx

    # creating new embedding (compleately untrained)
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
    # extracting the weigths of the original pretrained embeddign
    old_weights = original_embedding.weight
    new_weights = embedding.weight

    # replacing a chunk of the new parameters with te old parameters
    # to retain the ability to encode natrual language tokens
    embedding.weight = torch.nn.Parameter(
        torch.cat([old_weights.clone().to(device),
                   new_weights[original_embedding.num_embeddings:].clone().to(device)],
                  0))
    return embedding


def modify_embeddings(model: OPTForCausalLM, n_f_terms: int, device: str) -> OPTForCausalLM:
    original_embeddings = extract_embedding(model)
    new_embeddings = create_embedding(original_embeddings, n_f_terms, device)
    # Replacing the old embedding instance with the new embedding instance in the model instance
    model.set_input_embeddings(new_embeddings)
    return model


def load_pretrained_model(model_name: str, dtype: torch.dtype, tensor_parallel: bool, num_gpus: int, n_f_terms: int) -> OPTForCausalLM:
    """
    Loads a pretrained model in the OPT structure

    :return: OPTForCausalLM with pretrained weights
    """
    if num_gpus > 1:
        tensor_parallel = True

    # will probably never need a device map
    device_map=None

    # A dict to map the correct model urls
    HF_MAPPING = {
        "mini": ("facebook/galactica-125m", torch.float32),
        "base": ("facebook/galactica-1.3b", torch.float32),
        "standard": ("facebook/galactica-6.7b", torch.float32),
        "large": ("facebook/galactica-30b", torch.float32),
        "huge": ("facebook/galactica-120b", torch.float16)}

    # Analyzing the system (code by huggingface)
    max_memory = {}
    if num_gpus > 0 and not tensor_parallel:
        # based on https://github.com/huggingface/accelerate/blob/5315290b55ea9babd95a281a27c51d87b89d7c85/src/accelerate/utils/modeling.py#L274
        for i in range(num_gpus):
            _ = torch.tensor([0], device=i)
        for i in range(num_gpus):
            max_memory[i] = torch.cuda.mem_get_info(i)[0]
        device_map = "auto"
    max_memory["cpu"] = psutil.virtual_memory().available

    # Loading the model form web / from cache
    model = OPTForSequenceClassification.from_pretrained(HF_MAPPING[model_name][0], torch_dtype=dtype, low_cpu_mem_usage=True,
                                           device_map=device_map, max_memory=max_memory, num_labels=n_f_terms)

    return model
    

def load_and_modify_model(model_name: str,
                          dtype: torch.dtype,
                          tensor_parallel: bool,
                          num_gpus: int,
                          n_f_terms,
                          device: str) -> OPTForSequenceClassification:
    """
    This function is used to load the pretrained Galactica model and to modify it for Multilabel F-Term prediction
    
    model_name:      Name of the pretrained galactica instance. eg. 'mini' or 'huge'
    dtype:           Dtype in which the model should be loaded
    tensor_parallel: Indicates if parallelisation should be used in this model (just for multi gpu)
    num_gpus:        Number of GPUs used for the model 
    n_f_terms:       Number of possible F-Terms the model can predict (Number of classes / labels)
    device:          Device on which the model should be loaded.
    """

    _model = load_pretrained_model(model_name, dtype, tensor_parallel, num_gpus, n_f_terms)
    # Changing some configurations of the model
    _model.config.problem_type = "multi_label_classification"
    _model.config.vocab_size = n_f_terms + 50000
    #_model.num_labels = n_f_terms
    # Creating a new classification head, because the initial classification head has only two classes
    #_model.score = torch.nn.Linear(_model.config.word_embed_proj_dim, _model.num_labels, bias=False).to(device)
    return _model

if __name__=='__main__':
    pass