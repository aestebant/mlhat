from river.datasets import base
import stream


class Bibtex(base.RemoteDataset):
    """
    This dataset is based on the data of the ECML/PKDD 2008 discovery challenge. It contains 7395 bibtex entries from the BibSonomy social bookmark and publication sharing system, annotated with a subset of the tags assigned by BibSonomy users.

    Ioannis Katakis, Grigorios Tsoumakas, and Ioannis Vlahavas. Multilabel Text Classification for Automated Tag Suggestion. In Proceedings of the ECML/PKDD 2008 Discovery Challenge, 2008.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=7395,
            n_features=1836,
            n_outputs=159,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Bibtex_Meka.zip",
            unpack=True,
            filename="Bibtex.arff",
            size=3_505_238,
        )
    
    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "TAG_2005",
                "TAG_2006",
                "TAG_2007",
                "TAG_agdetection",
                "TAG_algorithms",
                "TAG_amperometry",
                "TAG_analysis",
                "TAG_and",
                "TAG_annotation",
                "TAG_antibody",
                "TAG_apob",
                "TAG_architecture",
                "TAG_article",
                "TAG_bettasplendens",
                "TAG_bibteximport",
                "TAG_book",
                "TAG_children",
                "TAG_classification",
                "TAG_clustering",
                "TAG_cognition",
                "TAG_collaboration",
                "TAG_collaborative",
                "TAG_community",
                "TAG_competition",
                "TAG_complex",
                "TAG_complexity",
                "TAG_compounds",
                "TAG_computer",
                "TAG_computing",
                "TAG_concept",
                "TAG_context",
                "TAG_cortex",
                "TAG_critical",
                "TAG_data",
                "TAG_datamining",
                "TAG_date",
                "TAG_design",
                "TAG_development",
                "TAG_diffusion",
                "TAG_diplomathesis",
                "TAG_disability",
                "TAG_dynamics",
                "TAG_education",
                "TAG_elearning",
                "TAG_electrochemistry",
                "TAG_elisa",
                "TAG_empirical",
                "TAG_energy",
                "TAG_engineering",
                "TAG_epitope",
                "TAG_equation",
                "TAG_evaluation",
                "TAG_evolution",
                "TAG_fca",
                "TAG_folksonomy",
                "TAG_formal",
                "TAG_fornepomuk",
                "TAG_games",
                "TAG_granular",
                "TAG_graph",
                "TAG_hci",
                "TAG_homogeneous",
                "TAG_imaging",
                "TAG_immunoassay",
                "TAG_immunoelectrode",
                "TAG_immunosensor",
                "TAG_information",
                "TAG_informationretrieval",
                "TAG_kaldesignresearch",
                "TAG_kinetic",
                "TAG_knowledge",
                "TAG_knowledgemanagement",
                "TAG_langen",
                "TAG_language",
                "TAG_ldl",
                "TAG_learning",
                "TAG_liposome",
                "TAG_litreview",
                "TAG_logic",
                "TAG_maintenance",
                "TAG_management",
                "TAG_mapping",
                "TAG_marotzkiwinfried",
                "TAG_mathematics",
                "TAG_mathgamespatterns",
                "TAG_methodology",
                "TAG_metrics",
                "TAG_mining",
                "TAG_model",
                "TAG_modeling",
                "TAG_models",
                "TAG_molecular",
                "TAG_montecarlo",
                "TAG_myown",
                "TAG_narrative",
                "TAG_nepomuk",
                "TAG_network",
                "TAG_networks",
                "TAG_nlp",
                "TAG_nonequilibrium",
                "TAG_notag",
                "TAG_objectoriented",
                "TAG_of",
                "TAG_ontologies",
                "TAG_ontology",
                "TAG_pattern",
                "TAG_patterns",
                "TAG_phase",
                "TAG_physics",
                "TAG_process",
                "TAG_programming",
                "TAG_prolearn",
                "TAG_psycholinguistics",
                "TAG_quantum",
                "TAG_random",
                "TAG_rdf",
                "TAG_representation",
                "TAG_requirements",
                "TAG_research",
                "TAG_review",
                "TAG_science",
                "TAG_search",
                "TAG_semantic",
                "TAG_semantics",
                "TAG_semanticweb",
                "TAG_sequence",
                "TAG_simulation",
                "TAG_simulations",
                "TAG_sna",
                "TAG_social",
                "TAG_socialnets",
                "TAG_software",
                "TAG_spin",
                "TAG_statistics",
                "TAG_statphys23",
                "TAG_structure",
                "TAG_survey",
                "TAG_system",
                "TAG_systems",
                "TAG_tagging",
                "TAG_technology",
                "TAG_theory",
                "TAG_topic1",
                "TAG_topic10",
                "TAG_topic11",
                "TAG_topic2",
                "TAG_topic3",
                "TAG_topic4",
                "TAG_topic6",
                "TAG_topic7",
                "TAG_topic8",
                "TAG_topic9",
                "TAG_toread",
                "TAG_transition",
                "TAG_visual",
                "TAG_visualization",
                "TAG_web",
                "TAG_web20",
                "TAG_wiki",
            ]
        )