import yaml
import numpy as np

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['comandos']:
    inputs.append(command['input'].lower())
    outputs.append('{}\{}'.format(command['entity'], command['action']))

# Processar texto: palavras, caracteres, bytes, sub-palavaras
chars = set()

for ch in inputs + outputs:
    if ch not in chars:
        chars.add(ch)

# Mapear char em index
chr2idx = {}
idx2chr = {}

for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch

max_seq = max([len(x) for x in inputs])

print('Número de chars:', len(chars))
print('Maior sequência', max_seq)        

# Criar o dataset (números de exemplos, pelo tamanho da seq, número de caracteres)
input_data = np.zeros((len(inputs), max_seq, len(chars)), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(inputs):
        input_data[i, k, chr2idx[ch]] = 1.0


print(input_data[0]) 


#print(inputs)
#print(outputs)    