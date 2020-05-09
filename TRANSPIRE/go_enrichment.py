import pandas as pd
import numpy as np

from goatools import mapslim
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_go_basic_obo, download_ncbi_associations
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

from .utils import uniprot_mapping_service

class GOAnalyzer:
    '''Wrapper to make analysis with GOATOOLS less complex

    The GOAAnalyzer class creates a GOEnrichmentStudyNS object that can be used to run consecutive enrichment studies using the same background gene list.

    Attributes:
        IDs (pd.DataFrame): ncbi_geneIDs for the input background proteins
        alpha (float): significance cutoff for enrichment analyses
        obodag (dict): GO Dag stored as a dict
        species (str): species ID for the given analysis (e.g. '9606' for homo sapiens)
        study (goatools.goea.go_enrichment_ns.GOEnrichmentStudyNS): GOEnrichmentStudyNS object used for running enrichment studies

    '''

    def __init__(self, background_proteins, species = '9606', alpha = 0.05, method = 'fdr_bh'):
        '''Initialize GOAnalyzer

        Args:
            background_proteins(Union(list, np.ndarray)): List or array of background protein accession numbers
            species (str, optional): species for analysis, defaults to '9606' which corresponds to homo sapiens
            alpha (float, optional): significance cutoff level, defaults to 0.05
            method (str, optional): multiple hypothesis correction method, defaults to 'fdr_bh'
        
        options for 'method' include (from the GOATOOLS documentation):
            'bonferroni',     #  0) Bonferroni one-step correction
            'sidak',          #  1) Sidak one-step correction
            'holm-sidak',     #  2) Holm-Sidak step-down method using Sidak adjustments
            'holm',           #  3) Holm step-down method using Bonferroni adjustments
            'simes-hochberg', #  4) Simes-Hochberg step-up method  (independent)
            'hommel',         #  5) Hommel closed method based on Simes tests (non-negative)
            'fdr_bh',         #  6) FDR Benjamini/Hochberg  (non-negative)
            'fdr_by',         #  7) FDR Benjamini/Yekutieli (negative)
            'fdr_tsbh',       #  8) FDR 2-stage Benjamini-Hochberg (non-negative)
            'fdr_tsbky',      #  9) FDR 2-stage Benjamini-Krieger-Yekutieli (non-negative)
            'fdr_gbs',        # 10) FDR adaptive Gavrilov-Benjamini-Sarkar
        '''

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
        '''Perform an enrichment analysis on the query_proteins

        Args:
            query_proteins (Union(list, np.ndarray)): List of protein accession numbers assess for functional enrichment
            return_all (bool, optional): If False (default), return only significantly-enriched GO terms (e.g. adj p-value <= GOAnalyzer.alpha). 
                                         Otherwise, if True, return all associated GO terms (including those that are not significant)
        
        Returns:
            results (pd.DataFrame): Results from GO enrichment analysis.

        '''

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
        '''Leverages GOATOOLS map_to_slim function to map GO terms to their GO-slim counterparts
        
        Args:
            GO_terms (Union(list, np.ndarray)): GO accession numbers to be mapped to slim terms
            return_all (bool, optional): Whether to return all, recusively-associated GO-slim terms for each given GO term (True) or only return direct descendents (False)

        Returns:
            result (dict): Dict pairs of GO accession (key) and its associated list of GO-slim terms (value) 
            
        '''

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