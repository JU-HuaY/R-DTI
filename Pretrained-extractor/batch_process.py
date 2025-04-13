from transformers import AutoModel, AutoTokenizer
# from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
prot_tokenizer = AutoTokenizer.from_pretrained("pre_trained_model/prot_bert_bfd", do_lower_case=False)
prot_model = AutoModel.from_pretrained("pre_trained_model/prot_bert_bfd").to(device)

# prot_tokenizer = T5Tokenizer.from_pretrained('pre_trained_model/prot_t5_xl_uniref50', do_lower_case=False)
# prot_model = T5EncoderModel.from_pretrained("pre_trained_model/prot_t5_xl_uniref50").to(device)
# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
prot_model.to(torch.float32)
# sequence_examples = ["PRTEINO", "SEQWENCE"]
sequences = "PRTEINO"
sequences2 = "PRTEINOAOTEOA"
protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding='do_not_pad')#"longest", max_length=1200, truncation=True, return_tensors='pt')
p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
# prepare your protein sequences as a list

with torch.no_grad():
    prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
print(len(sequences))
print(prot_feature.shape)
