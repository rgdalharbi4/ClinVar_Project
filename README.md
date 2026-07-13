ClinVar Analysis Project
Step 2: Problem Definition
1. What is the dataset about?
This dataset originates from the ClinVar database and focuses on conflicting clinical interpretations of genetic variants. It captures the complex nature of genomic medicine, where different clinical laboratories may provide different interpretations (e.g., "Pathogenic" vs. "Benign") for the same genetic variant. The data includes genomic features, variant classifications, and the degree of interpretation conflict, serving as a basis for understanding the challenges in genomic clinical diagnostic consistency.
2. What main questions will you explore?
Conflict Drivers: Which genomic features (e.g., variant type, size, location) are most strongly associated with conflicting interpretations?
Interpretation Patterns: Is there a specific trend in how laboratories classify variants, and are certain types of variants (e.g., missense vs. nonsense) more prone to "interpretation disagreement"?
Clinical Significance: How does the frequency of these variants correlate with the severity or consensus of their clinical interpretation?
3. What insights do you expect to uncover?
Predictive Indicators: I expect to identify specific genomic markers or "warning signs" that predict whether a genetic variant is likely to result in a conflicting interpretation before it is even classified.
Laboratory Discrepancy Map: An insight into whether discrepancies are distributed uniformly across the genome or concentrated in specific high-complexity regions.
Recommendation Potential: Understanding the root causes of these conflicts could lead to insights on how to improve clinical diagnostic standards, providing a "data-driven" rationale for why certain interpretations reach a consensus while others remain contested.

### Target Variable Definition (CLASS)
Based on the dataset documentation:
- **Class 0 (Concordant):** Represents variants where clinical laboratories are in agreement regarding the classification.
- **Class 1 (Conflicting):** Represents variants where clinical laboratories have provided conflicting classifications.

## Analysis Findings

Our exploratory data analysis of the ClinVar variant classifications provided several key insights:

* **Data Cleaning Strategy:** We identified extensive missing data in over 80% of specific genomic features. Consequently, we implemented a data-cleaning strategy to drop these irrelevant columns, ensuring a cleaner dataset for training.
* **Target Distribution:** The dataset presents an imbalanced distribution (75% Concordant vs. 25% Conflicting). We will account for this during model training using class-balancing techniques to ensure fair predictive performance.
* **Feature Relationships:** Analysis of feature correlations identified redundancy among certain numerical genomic features. These insights guide our feature selection process to focus on the most impactful variables for classification.


* **Feature Engineering & Data Preparation Plan
1. Keep & Scale (Numerical Features)
These are critical predictive features. They should be kept as-is and scaled (e.g., using StandardScaler) because they measure the functional impact and population frequency, which are key sources of disagreement among labs.
AF_ESP, AF_EXAC, AF_TGP: Population allele frequencies. Disagreement often arises when labs use different thresholds or datasets for variant rarity.
CADD_PHRED, CADD_RAW: Combined Annotation Dependent Depletion scores. These are the most powerful predictors for variant pathogenicity and a major source of interpretational conflict.
LoFtool: Loss-of-function intolerance score. Helps the model identify genes prone to conflict due to their functional complexity.
cDNA_position, CDS_position, Protein_position: Genomic and protein-level locations. Used as numerical features to detect if conflicts occur in specific, technically challenging regions of a gene.
2. Encode (Categorical Features)
These require Label or Frequency Encoding to convert them into a numerical format suitable for machine learning.
CHROM: Chromosome number. Maps the variant's location; converted to numeric to help the model identify "hotspots" of conflict.
IMPACT: Impact level (HIGH/MODERATE/LOW). Crucial; many conflicts occur because labs disagree on the severity of the impact.
SYMBOL: Gene symbol. Labs often have different levels of expertise/data for specific genes, leading to conflicts.
ORIGIN: Origin of the variant (e.g., germline, somatic). Important for identifying context-specific interpretations.
BIOTYPE: Biological type of the feature. Used to distinguish between coding and non-coding variants which undergo different evaluation protocols.
STRAND: Genomic strand. A technical feature that ensures correct mapping.
3. Handle with Caution (Clinical & Structural Features)
These contain rich information but are high-cardinality or prone to leakage.
CLNDN: Clinical disease name. High-cardinality; suggests using Frequency Encoding to capture which diseases are more "controversial" than others.
CLNVC, Feature_type, Feature: Variant type and structure. Important for understanding technical disagreement types.
MC (Molecular Consequence): Describes the mutation effect. Essential for characterizing the biological nature of the conflict.
Amino_acids, Codons: Sequence-level details. Converted to numeric to help model the complexity of the protein change.
4. Drop (Noise/Irrelevant Features)
These are dropped to prevent overfitting and remove data that does not contribute to predicting label conflict.
POS, REF, ALT: Spatial coordinates/identifiers. Without gene-specific context, they act as noise.
CLNHGVS: Clinical HGVS nomenclature. It is too specific (high-cardinality) and acts as an identifier rather than a predictive feature.
EXON: Often contains missing values or inconsistently formatted data across different genomic builds.
CLNDISDB, CLNDISDBINCL, CLNDNINCL: Clinical database identifiers. These provide redundant information or identifiers that are not predictive of conflict.
Allele: Mostly constant across rows and provides no discriminative power.
