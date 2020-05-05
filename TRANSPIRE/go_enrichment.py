import pandas as pd
import numpy as np

from goatools import mapslim
from goatools.associations import read_ncbi_gene2go
from goatools.base import download_go_basic_obo, download_ncbi_associations
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.obo_parser import GODag

from .utils import uniprot_mapping_service

class GOAnalyzer:
    def __init__(self, background_proteins, all_proteins, species = '9606', alpha = 0.05, method = 'fdr'):
        
        self.IDs = uniprot_mapping_service(background_proteins.values.tolist()+all_proteins[~all_proteins.isin(background_proteins)].values.tolist(), 'geneID')
        self.alpha = alpha

        background_IDs = self.IDs['GeneID'].astype(int).tolist()
        
        download_go_basic_obo()        
        download_ncbi_associations()

        self.obodag = GODag("go-basic.obo")

        self.species = species
        geneid2gos = read_ncbi_gene2go('gene2go', taxids = [int(species)])
        
        self.study = GOEnrichmentStudy(background_IDs, geneid2gos, self.obodag, propagate_counts = False, alpha = alpha, methods = [method])

    def get_enrichment(self, query_proteins, return_all = False):
        
        ids = self.IDs.loc[query_proteins, 'GeneID'].dropna().astype(int).tolist()

        if len(ids)>0:
            results = self.study.run_study(ids)

            # select entries above significance cutoff
            if not return_all:
                results = [r for r in results if r.get_pvalue() < self.alpha]

                if not len(results) > 0:
                    results = None

            # turn result entries into a dataframe
            if results is not None:

                fields = results[0].get_prtflds_default()
                results = {r.get_field_values(fields)[0]: r.get_field_values(fields)[1:] for r in results}
                results = pd.DataFrame.from_dict(results, orient = 'index')
                results.columns = fields[1:]
                results.index.names = ['GO accession']

                # only return enriched terms
                results = results[results['enrichment']=='e']

            return results

    def slim(self, GO_terms, return_all = False):
        '''Leverages GOAtools map_to_slim function to map GO terms to their slimmed counterparts'''

        # download slim obo file if it hasn't been downloaded already
        download_go_basic_obo(obo = 'goslim_generic.obo')
        
        slimdag = GODag('goslim_generic.obo')
        
        if len(GO_terms) < 1:
            raise ValueError('Length of GO_terms is less than 1')
        
        # make sure there are no redundant go terms
        GO_terms = np.unique(GO_terms)
        
        # map terms to direct and all slimmed decendents . . . yields GO accessions for slimmed terms
        result = {term: mapslim.mapslim(term, self.obodag, slimdag) for term in GO_terms}
        
        # map slimmed accessions to their respective GO terms
        for term in result:
            direct, _all = result[term]
            direct_terms = [slimdag.query_term(acc).name for acc in direct]

            if return_all == True:
                _all_terms = [slimdag.query_term(acc).name for acc in _all]
            else:
                _all_terms = []

            result[term] = [direct_terms, _all_terms]
            
        return result