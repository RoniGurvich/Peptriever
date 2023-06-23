---
{{  card_data  }}
datasets:
- {{  dataset  }}
---
# Protein BiEncoder Bert Model

Usage
```python
tokenizer = AutoTokenizer.from_pretrained("{{  model_repo  }}")
model = BiEncoder.from_pretrained("{{  model_repo  }}")
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

Model checkpint: `{{ model_id }}`