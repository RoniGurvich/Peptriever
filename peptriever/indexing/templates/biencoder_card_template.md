---
{{  card_data  }}
datasets:
- {{  dataset  }}
---
## Peptriever: A Bi-Encoder for large-scale protein-peptide binding search

For training details see our [Application Note](https://www.biorxiv.org/content/10.1101/2023.07.13.548811v1). 

Training code can be found in our [Github repo](https://github.com/RoniGurvich/Peptriever). 

A live demo is available on our [application page](https://peptriever.app) 

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{  model_repo  }}")
model = AutoModel.from_pretrained("{{  model_repo  }}", trust_remote_code=True)
model.eval()

peptide_sequence = "AAA"
protein_sequence = "MMM"
encoded_peptide = tokenizer.encode_plus(peptide_sequence, return_tensors='pt')
encoded_protein = tokenizer.encode_plus(protein_sequence, return_tensors='pt')

with torch.no_grad():
    peptide_output = model.forward1(encoded_peptide)
    protein_output = model.forward2(encoded_protein)

print("distance: ", torch.norm(peptide_output - protein_output, p=2))
```

## Version
Model checkpint: `{{ model_id }}`