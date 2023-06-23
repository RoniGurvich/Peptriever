from pathlib import Path
from typing import Set

from peptriever.data_sources.binding.binding_test_set import read_excluded_pdbs
from peptriever.data_sources.local_pdb import get_pdb_to_gene
from peptriever.data_sources.protein_transformations import protein_train_part
from peptriever.pdb_seq_lookup import get_pdb_lookup


class TrainParitioner:
    def __init__(
        self,
        binding_data_path: Path,
        pdb_path: Path,
        val_ratio: float,
        test_ratio: float,
        pdb_hf_repo: str,
    ):
        self.excluded_protein_names = read_excluded_pdbs(binding_data_path)
        self.pdb_lookup = get_pdb_lookup(pdb_hf_repo)
        self.excluded_sequences = self.read_excluded_sequences()
        self.pdb_to_gene = get_pdb_to_gene(pdb_dir=pdb_path)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def get_partition(self, pdb_id, peptide_chain, protein_chain):
        should_exclude = self._test_set_exclude_logic(
            pdb_id, peptide_chain, protein_chain
        )
        if should_exclude:
            train_part = "test"
        else:
            peptide_gene, protein_gene = self._get_genes(
                pdb_id, peptide_chain, protein_chain
            )
            gene_set = peptide_gene + protein_gene or None
            train_part = protein_train_part(
                gene_set or pdb_id, test_ratio=self.test_ratio, val_ratio=self.val_ratio
            )
        return train_part

    def _test_set_exclude_logic(self, pdb_id, peptide_chain, protein_chain):
        peptide_seq, protein_seq = self._get_seqs(pdb_id, peptide_chain, protein_chain)
        return any(
            [
                (pdb_id, protein_chain, pdb_id, peptide_chain)
                in self.excluded_protein_names,
                peptide_seq + protein_seq in self.excluded_sequences,
            ]
        )

    def _get_seqs(self, pdb_id, peptide_chain, protein_chain):
        peptide_seq = self.pdb_lookup.get((pdb_id, peptide_chain))
        protein_seq = self.pdb_lookup.get((pdb_id, protein_chain))
        return peptide_seq or "", protein_seq or ""

    def _get_genes(self, pdb_id, peptide_chain, protein_chain):
        peptide_gene = self.pdb_to_gene.get((pdb_id, peptide_chain), [""])[0]
        protein_gene = self.pdb_to_gene.get((pdb_id, protein_chain), [""])[0]
        return peptide_gene, protein_gene

    def read_excluded_sequences(self) -> Set[str]:
        excluded_sequences = set()
        for (
            prot_pdb_id,
            prot_chain,
            pep_pdb_id,
            pep_chain,
        ) in self.excluded_protein_names:
            prot_seq = self.pdb_lookup.get((prot_pdb_id, prot_chain))
            pep_seq = self.pdb_lookup.get((pep_pdb_id, pep_chain))
            if prot_seq and pep_seq:
                excluded_sequences.update([pep_seq + prot_seq])
        return excluded_sequences
