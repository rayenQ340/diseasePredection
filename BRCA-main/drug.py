import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.Data import CodonTable
import py3Dmol
import requests
from stmol import showmol
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re
try:
    import RNA
    rnafold_available = True
except ImportError:
    rnafold_available = False

# --------------------------
# Drug Design Functions
# --------------------------
def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0

def mock_check_specificity(sequence):
    """Mock version for Streamlit (avoid real BLAST calls)"""
    gc = calculate_gc_content(sequence)
    return max(40, 100 - abs(gc - 50))  # Peak at 50% GC

def calculate_drug_efficiency(binding_affinity, specificity, stability, delivery):
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    params = np.array([
        min(binding_affinity, 100),
        min(specificity, 100),
        min(stability, 100),
        min(delivery, 100)
    ])
    return min(100, np.dot(weights, params))

# Drug parameter space (20 variants)
DRUG_VARIANTS = [
    # ASO variants (5)
    {"type": "ASO", "mod": "Phosphorothioate", "stab": 80, "delivery": "Nanoparticle", "deliv_score": 70},
    {"type": "ASO", "mod": "2'-MOE", "stab": 85, "delivery": "GalNAc", "deliv_score": 75},
    {"type": "ASO", "mod": "LNA", "stab": 90, "delivery": "Liposome", "deliv_score": 65},
    {"type": "ASO", "mod": "PNA", "stab": 75, "delivery": "Exosome", "deliv_score": 60},
    {"type": "ASO", "mod": "PMO", "stab": 88, "delivery": "Polymer", "deliv_score": 72},
    
    # siRNA variants (5)
    {"type": "siRNA", "mod": "2'-OMe", "stab": 85, "delivery": "LNP", "deliv_score": 90},
    {"type": "siRNA", "mod": "2'-F", "stab": 88, "delivery": "GalNAc", "deliv_score": 92},
    {"type": "siRNA", "mod": "Unlocked", "stab": 82, "delivery": "Dendrimer", "deliv_score": 85},
    {"type": "siRNA", "mod": "Gapmer", "stab": 87, "delivery": "Peptide", "deliv_score": 80},
    {"type": "siRNA", "mod": "SNA", "stab": 90, "delivery": "Exosome", "deliv_score": 88},
    
    # CRISPR variants (5)
    {"type": "CRISPR", "mod": "gRNA", "stab": 75, "delivery": "AAV", "deliv_score": 65},
    {"type": "CRISPR", "mod": "sgRNA", "stab": 78, "delivery": "Lentivirus", "deliv_score": 70},
    {"type": "CRISPR", "mod": "xCas9", "stab": 80, "delivery": "Nanoparticle", "deliv_score": 75},
    {"type": "CRISPR", "mod": "HiFi", "stab": 82, "delivery": "Electroporation", "deliv_score": 60},
    {"type": "CRISPR", "mod": "BaseEdit", "stab": 85, "delivery": "VLP", "deliv_score": 68},
    
    # mRNA variants (3)
    {"type": "mRNA", "mod": "5'Cap", "stab": 90, "delivery": "LNP", "deliv_score": 95},
    {"type": "mRNA", "mod": "Pseudouridine", "stab": 92, "delivery": "Polymer", "deliv_score": 90},
    {"type": "mRNA", "mod": "Circular", "stab": 95, "delivery": "Dendrimer", "deliv_score": 88},
    
    # Aptamer variants (2)
    {"type": "Aptamer", "mod": "2'-F", "stab": 80, "delivery": "PEG", "deliv_score": 70},
    {"type": "Aptamer", "mod": "Spiegelmer", "stab": 85, "delivery": "Nanoparticle", "deliv_score": 75}
]

def design_drug_variant(variant, target_sequence):
    try:
        if variant["type"] == "ASO":
            target_mrna = target_sequence.transcribe()
            if len(target_mrna) < 20:
                return None
            seq = target_mrna[:20].reverse_complement()
            return {
                "type": f"ASO ({variant['mod']})",
                "sequence": str(seq),
                "efficiency": calculate_drug_efficiency(
                    calculate_gc_content(seq),
                    mock_check_specificity(seq),
                    variant["stab"],
                    variant["deliv_score"]
                ),
                "details": variant
            }
        
        elif variant["type"] == "siRNA":
            target_mrna = target_sequence.transcribe()
            if len(target_mrna) < 21:
                return None
            antisense = target_mrna[:21].reverse_complement()
            return {
                "type": f"siRNA ({variant['mod']})",
                "antisense": str(antisense),
                "sense": str(target_mrna[:21]),
                "efficiency": calculate_drug_efficiency(
                    calculate_gc_content(antisense),
                    mock_check_specificity(antisense),
                    variant["stab"],
                    variant["deliv_score"]
                ),
                "details": variant
            }
        
        elif variant["type"] == "CRISPR":
            pam_sites = [m.start() for m in re.compile(r"(?=(.{20}GG))").finditer(str(target_sequence))]
            if not pam_sites:
                return None
            grna = target_sequence[pam_sites[0]:pam_sites[0]+20]
            return {
                "type": f"CRISPR ({variant['mod']})",
                "gRNA": str(grna),
                "efficiency": calculate_drug_efficiency(
                    calculate_gc_content(grna),
                    mock_check_specificity(grna),
                    variant["stab"],
                    variant["deliv_score"]
                ),
                "details": variant
            }
        
        elif variant["type"] == "mRNA":
            return {
                "type": f"mRNA ({variant['mod']})",
                "efficiency": calculate_drug_efficiency(
                    80, 90, variant["stab"], variant["deliv_score"]
                ),
                "details": variant
            }
        
        elif variant["type"] == "Aptamer":
            seq = Seq("ACGU" * 10)
            mfe = -8.0 if not rnafold_available else RNA.fold_compound(str(seq)).mfe()[1]
            return {
                "type": f"Aptamer ({variant['mod']})",
                "efficiency": calculate_drug_efficiency(
                    min(abs(mfe) * 5, 100),
                    mock_check_specificity(seq),
                    variant["stab"],
                    variant["deliv_score"]
                ),
                "details": variant
            }
    
    except Exception:
        return None

# --------------------------
# Protein Structure Visualization
# --------------------------
def render_mol(pdb_id):
    view = py3Dmol.view(width=600, height=400)
    view.addModel(requests.get(f'https://files.rcsb.org/view/{pdb_id}.pdb').text, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    showmol(view, height=400, width=600)

def predict_structure(protein_seq):
    pdb_map = {
        "MGHHHHHH": "1CRN",
        "MGSDKIHH": "1L2Y",
        "MGSKGEEL": "2YGB",
        "M": "1UBQ"
    }
    return pdb_map.get(protein_seq[:8], "1UBQ")

# --------------------------
# Genomics Analysis
# --------------------------
class GenomicsAnalyzer:
    def __init__(self, sequence):
        self.sequence = sequence.upper().strip()
        self.dna_seq = Seq(self.sequence)
        
    def validate(self):
        return all(c in {'A', 'T', 'C', 'G'} for c in self.sequence)
    
    def get_stats(self):
        return {
            'length': len(self.dna_seq),
            'gc_content': gc_fraction(self.sequence) * 100,
            'at_content': 100 - (gc_fraction(self.sequence) * 100)
        }
    
    def transcribe(self):
        return str(self.dna_seq.transcribe())
    
    def translate(self):
        return str(self.dna_seq.translate(to_stop=True))
    
    def get_3d_structure(self):
        protein_seq = self.translate()
        pdb_id = predict_structure(protein_seq)
        return pdb_id, protein_seq

# --------------------------
# Clinical Predictor
# --------------------------
class ClinicalPredictor:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        self.model = RandomForestClassifier(n_estimators=50)
        self._train_model()
    
    def _train_model(self):
        sequences = ["ATGC"*25, "GCAT"*25, "TTAA"*25, "AATT"*25]
        labels = [1, 1, 0, 0]
        X = self.vectorizer.fit_transform(sequences)
        self.model.fit(X, labels)
    
    def predict_er_status(self, dna_seq):
        X = self.vectorizer.transform([dna_seq])
        return "Positive" if self.model.predict(X)[0] == 1 else "Negative"

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(page_title="BRCA 3D Analyzer", layout="wide")
    st.title("🧬 Breast Cancer Digital Twin with Therapeutic Design")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Input Parameters")
        dna_input = st.text_area(
            "Enter DNA Sequence",
            "ATGCGATACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",
            height=200
        )
        analyze_btn = st.button("Analyze Sequence")
        design_btn = st.button("Design Therapeutics")
    
    if analyze_btn or design_btn:
        if len(dna_input) < 20:
            st.error("Please enter a valid DNA sequence (minimum 20 bases)")
        else:
            try:
                analyzer = GenomicsAnalyzer(dna_input)
                predictor = ClinicalPredictor()
                
                if not analyzer.validate():
                    st.error("Invalid DNA sequence - only A,T,C,G allowed")
                    return
                
                if analyze_btn:
                    with st.spinner("Processing genomic data..."):
                        stats = analyzer.get_stats()
                        er_status = predictor.predict_er_status(dna_input)
                        pdb_id, protein_seq = analyzer.get_3d_structure()
                        mutations = {
                            'ER_status': er_status,
                            'HER2': 'Amplified',
                            'PIK3CA': 'H1047R'
                        }
                    
                    st.success("Analysis Complete")
                    
                    tab1, tab2, tab3 = st.tabs(["Genomic Analysis", "Protein Translation", "3D Structure"])
                    
                    with tab1:
                        st.subheader("Sequence Statistics")
                        st.json(stats)
                        
                        st.subheader("Nucleotide Composition")
                        fig = px.bar(
                            pd.DataFrame({
                                'Base': ['A', 'T', 'C', 'G'],
                                'Count': [dna_input.count(b) for b in ['A', 'T', 'C', 'G']]
                            }),
                            x='Base', y='Count'
                        )
                        st.plotly_chart(fig)
                        
                        st.subheader("Clinical Findings")
                        st.json(mutations)
                    
                    with tab2:
                        st.subheader("RNA Transcription")
                        st.code(analyzer.transcribe())
                        
                        st.subheader("Protein Translation")
                        st.code(protein_seq)
                        
                        st.subheader("Amino Acid Composition")
                        aa_counts = pd.DataFrame({
                            'Amino Acid': list(set(protein_seq)),
                            'Count': [protein_seq.count(aa) for aa in set(protein_seq)]
                        })
                        st.plotly_chart(px.bar(aa_counts, x='Amino Acid', y='Count'))
                    
                    with tab3:
                        st.subheader("Predicted 3D Protein Structure")
                        st.info(f"Displaying structure for PDB ID: {pdb_id} (example)")
                        render_mol(pdb_id)
                
                if design_btn:
                    with st.spinner("Designing 20 therapeutic variants..."):
                        drugs = [design_drug_variant(v, analyzer.dna_seq) for v in DRUG_VARIANTS]
                        valid_drugs = [d for d in drugs if d is not None]
                        
                        if not valid_drugs:
                            st.error("No valid drugs designed")
                            return
                            
                        best_drug = max(valid_drugs, key=lambda x: x["efficiency"])
                    
                    st.success("Therapeutic Design Complete")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏆 Optimal Therapeutic")
                        st.metric("Drug Type", best_drug["type"])
                        st.metric("Efficiency Score", f"{best_drug['efficiency']:.1f}%")
                        
                        if "sequence" in best_drug:
                            st.text_area("Sequence", best_drug["sequence"], height=100)
                        elif "gRNA" in best_drug:
                            st.text_area("gRNA", best_drug["gRNA"], height=100)
                        
                        st.write("**Modifications:**", best_drug["details"]["mod"])
                        st.write("**Delivery:**", best_drug["details"]["delivery"])
                    
                    with col2:
                        st.subheader("Efficiency Components")
                        eff_data = {
                            "Parameter": ["Binding", "Specificity", "Stability", "Delivery"],
                            "Score": [
                                best_drug["details"].get("binding_affinity", 80),
                                best_drug["details"].get("specificity", 90),
                                best_drug["details"]["stab"],
                                best_drug["details"]["deliv_score"]
                            ]
                        }
                        st.bar_chart(pd.DataFrame(eff_data).set_index("Parameter"))
                        
                        st.write("**All Variants Tested:**")
                        st.dataframe(
                            pd.DataFrame([{
                                "Type": d["type"],
                                "Efficiency": d["efficiency"],
                                "Stability": d["details"]["stab"],
                                "Delivery": d["details"]["deliv_score"]
                            } for d in valid_drugs]).sort_values("Efficiency", ascending=False),
                            height=300
                        )
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()