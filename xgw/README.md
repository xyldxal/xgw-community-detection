# A supervised co-complex probability weighting of yeast composite protein networks using gradient-boosted trees for protein complex detection

## Data

### Databases. (located in `data/databases` and `data/swc`):

These are the raw and external database files used in this study.

- Original composite protein network. (`data/swc/data_yeast.txt`)
  - This is the yeast composite protein network used in SWC [[12]](#12).
  - This has the following features: TOPO, TOPO_L2, STRING, and CO_OCCUR.
  - Link: https://www.comp.nus.edu.sg/~wongls/projects/complexprediction/SWC-31oct14/
- Gene co-expression data (GSE3431 [[10]](#10)). (`data/databases/GSE3431_setA_family.pcl`)
  - GSE3431 was used, which was downloaded from the SGD website.
  - Link: http://sgd-archive.yeastgenome.org/expression/microarray/Tu_2005_PMID_16254148/ (`GSE3431_setA_family.pcl`)
- Gene ontology [[1]](#1)[[3]](#3). (`data/databases/gene_ontology.obo` )
  - Version 2011-10-31 was used, which was downloaded from the GO website.
  - Link: http://release.geneontology.org/2011-11-01/ontology/index.html (`gene_ontology.obo`)
- GO annotations [[2]](#2). (`data/databases/sgd.gaf`)
  - Version 2011-10-29 was used, which was downloaded from the GO website and was provided by SGD.
  - Link: http://release.geneontology.org/2011-11-01/annotations/index.html (`sgd.gaf.gz`)
- iRefIndex [[8]](#8). (`data/databases/large/irefindex 559292 mitab26.zip`)
  - Version 19.0 (2022-08-22) was used, which was downloaded from the iRefIndex website.
  - **Note**: Only the publication entries before 2012 were selected from this database.
  - **Note**: Due to the large size of the iRefIndex database, it is zipped so you need to extract it.
  - Link: https://irefindex.vib.be/wiki/index.php/README_MITAB2.6_for_iRefIndex_19.0
- DIP PPIN [[9]](#9). (`data/databases/Scere20170205.txt`)
  - Version 2017-02-05 was used, which was downloaded from the DIP website.
  - Link: https://dip.doe-mbi.ucla.edu/dip/Main.cgi
- CYC2008 [[7]](#7). (`data/swc/complexes_CYC.txt`)
  - This was already provided by the SWC software package.

### Preprocessed Data. (`data/preprocessed`)

Preprocessed data can be found here. The cross-validation splits used by both XGW and SWC can also be found here (`data/preprocessed/cross_val_table.csv` and `data/preprocessed/dip_cross_val_table.csv`).

### Clusters. (`data/clusters`)

This is where the MCL algorithm software outputs its predicted clusters (after running `src/run_mcl.sh`, see the next sections).

### Evaluations. (`data/evals`)

This is where evaluation data are stored (precision-recall AUC, log loss, Brier score loss, etc...)

### Scores. (`data/scores`)

This is where the feature scores are stored (TOPO, TOPO_L2, STRING, CO_OCCUR, REL, CO_EXP, GO_CC, GO_BP, GO_MF).

### SWC data. (`data/swc`)

This is where the data that are provided and/or needed by SWC are stored.

### Training (`data/training`)

Hyperparameter settings are stored here based on previous training, as well as the computed feature importances.

### Weighted (`data/weighted`)

The weighted protein networks computed from all the 19 weighting methods.

## Source code. (`src/`)

All the source codes of XGW are stored here.

`preprocessor.py` - Preprocesses data in `data/databases` and stores preprocessed data to `data/preprocessed`. This is run only once (if `data/preprocessed`) is empty.

`dip_preprocessor.py` - Preprocesses raw DIP PPIN and constructs the base composite network for this network by topologically weighting it (TOPO and TOPO_L2), and integrating STRING and CO_OCCUR features.

`co_exp_scoring.py` and `rel_scoring.py` - Scores the Original and DIP composite network based on CO_EXP and REL.

`weighting.py` - Weights the two composite networks using the 19 weighting methods. Also, outputs the feature importances of XGW to `data/training`. Running this script takes a lot of time. If you want to rerun this script yourself, change the `re_weight` variable to `True` in the `main` method call. For more information, see the comments in this file.

`evaluate_comp_edges.py` - This evaluates the co-complex edge classification performance of each method. If you want to rerun this script yourself, change the `re_eval` variable to `True` in the `main` method call. For more information, see the notes in the comments of this file.

`evaluate_clusters.py` - This evaluates the cluster prediction performance of each method. If you want to rerun this script yourself, change the `re_eval` variable to `True` in the `main` method call. For more information, see the notes in the comments of this file.

`visualize_results.py` - This outputs all the tables and figures included in the paper.

The following sequence is how the scripts were run in order from the very start to the very end, assuming that `data/databases` were already given.

`preprocessor.py` > `dip_preprocessor.py` > `co_exp_scoring.py` > `rel_scoring.py` > TCSS (see next section) > SWC (see next section) > `weighting.py` > `evaluate_comp_edges.py` > `evaluate_clusters.py` > `visualize_results.py`.

## Results and Supplementary Materials

Results (figures) can be found in `results/`. These graphs were derived from performance evaluations in `data/evals/` and `data/training/` (for the feature importances). Moreover, a supplementary material document is also provided here (`SupplementaryMaterial.pdf`).

## External Software packages/services used

### TCSS. (`TCSS/`)

This study uses the Topological Clustering Semantic Similarity (TCSS) [[6]](#6) software package proposed by Jain & Bader (2010) on their study: _An improved method for scoring protein-protein
interactions using semantic similarity within the gene ontology._

Requirements: Python.

Link: https://baderlab.org/Software/TCSS

TCSS is licensed under GNU Lesser General Public License v3.0. The said license can be found in the `TCSS/` directory. For more information, the readers are directed to `TCSS/README.md`.

Commands to run TCSS:

For the Original composite network:

`python TCSS/tcss.py -i data/preprocessed/swc_edges.csv -o data/scores/go_ss_scores.csv --drop="IEA" --gene=data/databases/sgd.gaf --go=data/databases/gene_ontology.obo`

to generate the GO-weighted network (`data/scores/go_ss_scores.csv`) used in the study.

For the DIP composite network, the following command was used:

`python TCSS/tcss.py -i data/preprocessed/dip_edges.csv -o data/scores/dip_go_ss_scores.csv --drop="IEA" --gene=data/databases/sgd.gaf --go=data/databases/gene_ontology.obo`

### SWC

This study also uses the SWC [[12]](#12) software package and source files. The SWC method was proposed by Yong et. al. (2012) on their study: _Supervised maximum-likelihood weighting of composite protein networks for complex prediction_. We would like to thank the authors of SWC, the main inspiration for this study, for permitting us to use their software and data.

Requirements: Perl.

Link: https://www.comp.nus.edu.sg/~wongls/projects/complexprediction/SWC-31oct14/

Commands to run SWC:

For the Original composite network:

`perl score_edges.pl -i data_yeast.txt -c complexes_CYC.txt -m x -x cross_val.csv -e 0 -o "swc"`

For the DIP composite network:

`perl score_edges.pl -i dip_data_yeast.txt -c complexes_CYC.txt -m x -x dip_cross_val.csv -e 0 -o "dip_swc"`

### MCL

The Markov Cluster (MCL) [[4]](#4)[[5]](#5)[[11]](#11) Algorithm was used to cluster the weighted protein networks.

Requirements: MCL.

Link: https://github.com/micans/mcl

To run MCL (assuming MCL is installed in `~/local/bin/mcl`), navigate to `src`, then run:

`./run_mcl.sh ../data/weighted/ ./ 4`

This command will run the MCL algorithm on the weighted protein networks in `data/weighted` with inflation parameter set to 4. Note that the inflation parameter can be set to arbitrary value, preferrable around the range of [2, 5].

Running the above command will output `out.{file}.csv.I40` files to `data/clusters`.

If you want to set inflation parameter to 2, run:

`./run_mcl.sh ../data/weighted/ ./ 2`

which will output `out.{file}.csv.I20` files to `data/clusters`.

### UniProt ID Mapping

The UniProt [[13]](#13) Retrieve/ID mapping service was used to map each UniProtKB AC/ID in the DIP PPIN to its corresponding KEGG entry (systematic name). This was used to produce KEGG mapping file in `data/databases/dip_uniprot_kegg_mapped.tsv`.

Link: https://www.uniprot.org/id-mapping

## Internal software packages used

The following software/packages were used in building XGW:

- Python 3.11.0 [[14]](#14)
- Polars 0.18.4 [[15]](#15)
- scikit-learn 1.3.0 [[16]](#16)
- XGBoost 1.7.6 [[17]](#17)
- matplotlib 3.7.1 [[18]](#18)
- seaborn 0.12.2 [[19]](#19)
- numpy 1.25.0 [[20]](#20)

## Setup and Installation

To install the required Python packages for XGW, create a Python virtual environment, then run the following command after activating the environment:

`pip install -r requirements.txt`

Also, make sure that you have already downloaded the SWC software (as well as Perl) and the MCL algorithm software. You do not need to download TCSS from their website since the said software package is already included in this repository.

## References (for the data and software used)

<a id="1">[1]</a> M Ashburner, C A Ball, J A Blake, D Botstein, H Butler, J M Cherry, A P Davis, K Dolinski, S S Dwight, J T Eppig, M A Harris, D P Hill, L Issel-Tarver, A Kasarskis, S Lewis, J C Matese, J E Richardson, M Ringwald, G M Rubin, and G Sherlock. Gene ontology: tool for the unification of biology. the gene ontology consortium. Nat. Genet., 25(1):25–29, May 2000.

<a id="2">[2]</a> J Michael Cherry, Eurie L Hong, Craig Amundsen, Rama Balakrishnan, Gail Binkley, Esther T Chan, Karen R Christie, Maria C Costanzo, Selina S Dwight, Stacia R Engel, Dianna G Fisk,Jodi E Hirschman, Benjamin C Hitz, Kalpana Karra, Cynthia J Krieger, Stuart R Miyasato, Rob S Nash, Julie Park, Marek S Skrzypek, Matt Simison, Shuai Weng, and Edith D Wong. Saccharomyces genome database: the genomics resource of budding yeast. Nucleic Acids Res., 40(Database issue):D700–5, January 2012.

<a id="3">[3]</a> The Gene Ontology Consortium, Suzi A Aleksander, James Balhoff, Seth Carbon, J Michael Cherry, Harold J Drabkin, Dustin Ebert, Marc Feuermann, Pascale Gaudet, Nomi L Harris, David P Hill, Raymond Lee, Huaiyu Mi, Sierra Moxon, Christopher J Mungall, Anushya Muruganugan, Tremayne Mushayahama, Paul W Sternberg, Paul D Thomas, Kimberly Van Auken, Jolene Ramsey, Deborah A Siegele, Rex L Chisholm, Petra Fey, Maria Cristina Aspromonte, Maria Victoria Nugnes, Federica Quaglia, Silvio Tosatto, Michelle Giglio, Suvarna Nadendla, Giulia Antonazzo, Helen Attrill, Gil dos Santos, Steven Marygold, Victor Strelets, Christopher J Tabone, Jim Thurmond, Pinglei Zhou, Saadullah H Ahmed, Praoparn Asanitthong, Diana Luna Buitrago, Meltem N Erdol, Matthew C Gage, Mohamed Ali Kadhum, Kan Yan Chloe Li, Miao Long, Aleksandra Michalak, Angeline Pesala, Armalya Pritazahra, Shirin C C Saverimuttu, Renzhi Su, Kate E Thurlow, Ruth C Lovering, Colin Logie, Snezhana Oliferenko, Judith Blake, Karen Christie, Lori Corbani, Mary E Dolan, Harold J Drabkin, David P Hill, Li Ni, Dmitry Sitnikov, Cynthia Smith, Alayne Cuzick, James Seager, Laurel Cooper, Justin Elser, Pankaj Jaiswal, Parul Gupta, Pankaj Jaiswal, Sushma Naithani, Manuel Lera-Ramirez, Kim Rutherford, Valerie Wood, Jeffrey L De Pons, Melinda R Dwinell, G Thomas Hayman, Mary L Kaldunski, Anne E Kwitek, Stanley J F Laulederkind, Marek A Tutaj, Mahima Vedi, Shur-Jen Wang, Peter D’Eustachio, Lucila Aimo, Kristian Axelsen, Alan Bridge, Nevila Hyka-Nouspikel, Anne Morgat, Suzi A Aleksander, J Michael Cherry, Stacia R Engel, Kalpana Karra, Stuart R Miyasato, Robert S Nash, Marek S Skrzypek, Shuai Weng, Edith D Wong, Erika Bakker, Tanya Z Berardini, Leonore Reiser, Andrea Auchincloss, Kristian Axelsen, Ghislaine Argoud-Puy, Marie-Claude Blatter, Emmanuel Boutet, Lionel Breuza, Alan Bridge, Cristina Casals-Casas, Elisabeth Coudert, Anne Estreicher, Maria Livia Famiglietti, Marc Feuermann, Arnaud Gos, Nadine Gruaz-Gumowski, Chantal Hulo, Nevila Hyka-Nouspikel, Florence Jungo, Philippe Le Mercier, Damien Lieberherr, Patrick Masson, Anne Morgat, Ivo Pedruzzi, Lucille Pourcel, Sylvain Poux, Catherine Rivoire, Shyamala Sundaram, Alex Bateman, Emily Bowler-Barnett, Hema Bye-A-Jee, Paul Denny, Alexandr Ignatchenko, Rizwan Ishtiaq, Antonia Lock, Yvonne Lussi, Michele Magrane, Maria J Martin, Sandra Orchard, Pedro Raposo, Elena Speretta, Nidhi Tyagi, Kate Warner, Rossana Zaru,Alexander D Diehl, Raymond Lee, Juancarlos Chan,Stavros Diamantakis, Daniela Raciti, Mag-dalena Zarowiecki, Malcolm Fisher, Christina James-Zorn, Virgilio Ponferrada, Aaron Zorn, Sridhar Ramachandran, Leyla Ruzicka, and Monte Westerfield. The Gene Ontology knowledgebase in 2023. Genetics, 224(1):iyad031, 03 2023.

<a id="4">[4]</a> Stijn Dongen. Graph clustering by flow simulation. PhD thesis, Center for Math and Computer Science (CWI), 05 2000.

<a id="5">[5]</a> A. J. Enright, S. Van Dongen, and C. A. Ouzounis. An efficient algorithm for large-scale detection of protein families. Nucleic Acids Res, 30(7):1575–1584, Apr 2002.

<a id="6">[6]</a> Shobhit Jain and Gary D. Bader. An improved method for scoring protein-protein interactions using semantic similarity within the gene ontology. BMC Bioinformatics, 11(1):562, Nov 2010.

<a id="7">[7]</a> Shuye Pu, Jessica Wong, Brian Turner, Emerson Cho, and Shoshana J Wodak. Up-to-date catalogues of yeast protein complexes. Nucleic Acids Res., 37(3):825–831, February 2009.

<a id="8">[8]</a> Sabry Razick, George Magklaras, and Ian M Donaldson. iRefIndex: a consolidated protein interaction database with provenance. BMC Bioinformatics, 9:405, September 2008.

<a id="9">[9]</a> Lukasz Salwinski, Christopher S Miller, Adam J Smith, Frank K Pettit, James U Bowie, and David Eisenberg. The database of interacting proteins: 2004 update. Nucleic Acids Res., 32(Database issue):D449–51, January 2004.

<a id="10">[10]</a> B. P. Tu, A. Kudlicki, M. Rowicka, and S. L. McKnight. Logic of the yeast metabolic cycle: temporal compartmentalization of cellular processes. Science, 310(5751):1152–1158, Nov 2005.

<a id="11">[11]</a> Stijn Van Dongen. Graph clustering via a discrete uncoupling process. SIAM Journal on Matrix Analysis and Applications, 30(1):121–141, 2008.

<a id="12">[12]</a> C. H. Yong, G. Liu, H. N. Chua, and L. Wong. Supervised maximum-likelihood weighting of composite protein networks for complex prediction. BMC Syst Biol, 6 Suppl 2(Suppl 2):S13, 2012.

<a id="13">[13]</a> The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids
Research, 51(D1):D523–D531, 11 2022.

<a id="14">[14]</a> Python Software Foundation. Python Language Reference, version 3.11. Available at http://www.python.org

<a id="15">[15]</a> Ritchie Vink. Polars. In GitHub repository. GitHub. https://github.com/pola-rs/polars

<a id="16">[16]</a> F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

<a id="17">[17]</a> Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.

<a id="18">[18]</a> J. D. Hunter. Matplotlib: A 2d graphics environment. Computing in Science & Engineering, 9(3):90–95, 2007.

<a id="19">[19]</a> Michael L. Waskom. seaborn: statistical data visualization. Journal of Open Source Software, 6(60):3021, 2021.

<a id="20">[20]</a> Charles R. Harris, K. Jarrod Millman, St´efan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fern´andez del R´ıo, Mark Wiebe, Pearu Peterson, Pierre G´erard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature, 585(7825):357–362, September 2020.
