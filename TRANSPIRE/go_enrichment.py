import pandas as pd
import numpy as np

from goatools import mapslim
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_go_basic_obo, download_ncbi_associations
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

from .utils import uniprot_mapping_service

class GOAnalyzer:
    def __init__(self, background_proteins, species = '9606', alpha = 0.05, method = 'fdr_bh'):

        if isinstance(background_proteins, pd.Index) or isinstance(background_proteins, pd.Series):
            background_proteins = background_proteins.values.tolist()

        elif isinstance(background_proteins, np.ndarray):
            background_proteins = background_proteins.tolist()

        assert(isinstance(background_proteins, list))
        
        self.IDs = uniprot_mapping_service(background_proteins, 'geneID')
        self.alpha = alpha

        background_IDs = self.IDs['GeneID'].astype(int).tolist()
        
        obo_fname = download_go_basic_obo()        
        fin_gene2go = download_ncbi_associations()

        self.obodag = GODag("go-basic.obo")

        self.species = species

        geneid2gos = Gene2GoReader(fin_gene2go, taxids = [int(species)])
        ns2assoc = geneid2gos.get_ns2assc()
        
        self.study = GOEnrichmentStudyNS(background_IDs, ns2assoc, self.obodag, propagate_counts = False, alpha = alpha, methods = [method])

    def get_enrichment(self, query_proteins, return_all = False):
        
        ids = self.IDs.loc[query_proteins, 'GeneID'].dropna().astype(int).tolist()

        if len(ids)>0:
            results = self.study.run_study(ids)

            self.curr_results = results

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