import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, EState
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model_path = 'trained_model.sav'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to calculate all molecular descriptors
def calculate_all_descriptors(mol):
    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]
    descriptors = {}
    for desc_name, desc_func in Descriptors.descList:
        try:
            descriptors[desc_name] = desc_func(mol)
        except Exception as e:
            descriptors[desc_name] = str(e)
    return descriptors

# Add a sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Description", "Contact"])

# Home page
if page == "Home":
    st.title("Predicting Permeability of Cyclic Peptides")
    st.subheader("Calculating molecular descriptors from SMILES string")

    # User input: SMILES string
    smiles_input = st.text_input("Enter SMILES string:", "")

    if smiles_input:
        try:
            # Convert SMILES string to RDKit molecule object
            mol = Chem.MolFromSmiles(smiles_input)

            if mol:
                # Calculate all descriptors
                descriptors = calculate_all_descriptors(mol)

                # Convert descriptors to DataFrame
                df = pd.DataFrame(descriptors.items(), columns=["Descriptor", "Value"])

                # # Display descriptors
                # st.write("### Molecular Descriptors")
                # st.dataframe(df)

                # Prepare the input for the model using selected descriptors
                model_input = np.array([[
                    descriptors['MaxEStateIndex'],
                    descriptors['MinEStateIndex'],
                    descriptors['MaxAbsEStateIndex'],
                    descriptors['MinAbsEStateIndex'],
                    descriptors['qed'],
                    descriptors['MolWt'],
                    descriptors['HeavyAtomMolWt'],
                    descriptors['ExactMolWt'],
                    descriptors['NumValenceElectrons'],
                    descriptors['NumRadicalElectrons'],
                    descriptors['MaxPartialCharge'],
                    descriptors['MinPartialCharge'],
                    descriptors['MaxAbsPartialCharge'],
                    descriptors['MinAbsPartialCharge'],
                    descriptors['FpDensityMorgan1'],
                    descriptors['FpDensityMorgan2'],
                    descriptors['FpDensityMorgan3'],
                    descriptors['BCUT2D_MWHI'],
                    descriptors['BCUT2D_MWLOW'],
                    descriptors['BCUT2D_CHGHI'],
                    descriptors['BCUT2D_CHGLO'],
                    descriptors['BCUT2D_LOGPHI'],
                    descriptors['BCUT2D_LOGPLOW'],
                    descriptors['BCUT2D_MRHI'],
                    descriptors['BCUT2D_MRLOW'],
                    descriptors['BalabanJ'],
                    descriptors['BertzCT'],
                    descriptors['Chi0'],
                    descriptors['Chi0n'],
                    descriptors['Chi0v'],
                    descriptors['Chi1'],
                    descriptors['Chi1n'],
                    descriptors['Chi1v'],
                    descriptors['Chi2n'],
                    descriptors['Chi2v'],
                    descriptors['Chi3n'],
                    descriptors['Chi3v'],
                    descriptors['Chi4n'],
                    descriptors['Chi4v'],
                    descriptors['HallKierAlpha'],
                    descriptors['Ipc'],
                    descriptors['Kappa1'],
                    descriptors['Kappa2'],
                    descriptors['Kappa3'],
                    descriptors['LabuteASA'],
                    descriptors['PEOE_VSA1'],
                    descriptors['PEOE_VSA10'],
                    descriptors['PEOE_VSA11'],
                    descriptors['PEOE_VSA12'],
                    descriptors['PEOE_VSA13'],
                    descriptors['PEOE_VSA14'],
                    descriptors['PEOE_VSA2'],
                    descriptors['PEOE_VSA3'],
                    descriptors['PEOE_VSA4'],
                    descriptors['PEOE_VSA5'],
                    descriptors['PEOE_VSA6'],
                    descriptors['PEOE_VSA7'],
                    descriptors['PEOE_VSA8'],
                    descriptors['PEOE_VSA9'],
                    descriptors['SMR_VSA1'],
                    descriptors['SMR_VSA10'],
                    descriptors['SMR_VSA2'],
                    descriptors['SMR_VSA3'],
                    descriptors['SMR_VSA4'],
                    descriptors['SMR_VSA5'],
                    descriptors['SMR_VSA6'],
                    descriptors['SMR_VSA7'],
                    descriptors['SMR_VSA8'],
                    descriptors['SMR_VSA9'],
                    descriptors['SlogP_VSA1'],
                    descriptors['SlogP_VSA10'],
                    descriptors['SlogP_VSA11'],
                    descriptors['SlogP_VSA12'],
                    descriptors['SlogP_VSA2'],
                    descriptors['SlogP_VSA3'],
                    descriptors['SlogP_VSA4'],
                    descriptors['SlogP_VSA5'],
                    descriptors['SlogP_VSA6'],
                    descriptors['SlogP_VSA7'],
                    descriptors['SlogP_VSA8'],
                    descriptors['SlogP_VSA9'],
                    descriptors['TPSA'],
                    descriptors['EState_VSA1'],
                    descriptors['EState_VSA10'],
                    descriptors['EState_VSA11'],
                    descriptors['EState_VSA2'],
                    descriptors['EState_VSA3'],
                    descriptors['EState_VSA4'],
                    descriptors['EState_VSA5'],
                    descriptors['EState_VSA6'],
                    descriptors['EState_VSA7'],
                    descriptors['EState_VSA8'],
                    descriptors['EState_VSA9'],
                    descriptors['VSA_EState1'],
                    descriptors['VSA_EState10'],
                    descriptors['VSA_EState2'],
                    descriptors['VSA_EState3'],
                    descriptors['VSA_EState4'],
                    descriptors['VSA_EState5'],
                    descriptors['VSA_EState6'],
                    descriptors['VSA_EState7'],
                    descriptors['VSA_EState8'],
                    descriptors['VSA_EState9'],
                    descriptors['FractionCSP3'],
                    descriptors['HeavyAtomCount'],
                    descriptors['NHOHCount'],
                    descriptors['NOCount'],
                    descriptors['NumAliphaticCarbocycles'],
                    descriptors['NumAliphaticHeterocycles'],
                    descriptors['NumAliphaticRings'],
                    descriptors['NumAromaticCarbocycles'],
                    descriptors['NumAromaticHeterocycles'],
                    descriptors['NumAromaticRings'],
                    descriptors['NumHAcceptors'],
                    descriptors['NumHDonors'],
                    descriptors['NumHeteroatoms'],
                    descriptors['NumRotatableBonds'],
                    descriptors['NumSaturatedCarbocycles'],
                    descriptors['NumSaturatedHeterocycles'],
                    descriptors['NumSaturatedRings'],
                    descriptors['RingCount'],
                    descriptors['MolLogP'],
                    descriptors['MolMR'],
                    descriptors['fr_Al_COO'],
                    descriptors['fr_Al_OH'],
                    descriptors['fr_Al_OH_noTert'],
                    descriptors['fr_ArN'],
                    descriptors['fr_Ar_COO'],
                    descriptors['fr_Ar_N'],
                    descriptors['fr_Ar_NH'],
                    descriptors['fr_Ar_OH'],
                    descriptors['fr_COO'],
                    descriptors['fr_COO2'],
                    descriptors['fr_C_O'],
                    descriptors['fr_C_O_noCOO'],
                    descriptors['fr_C_S'],
                    descriptors['fr_HOCCN'],
                    descriptors['fr_Imine'],
                    descriptors['fr_NH0'],
                    descriptors['fr_NH1'],
                    descriptors['fr_NH2'],
                    descriptors['fr_N_O'],
                    descriptors['fr_Ndealkylation1'],
                    descriptors['fr_Ndealkylation2'],
                    descriptors['fr_Nhpyrrole'],
                    descriptors['fr_SH'],
                    descriptors['fr_aldehyde'],
                    descriptors['fr_alkyl_carbamate'],
                    descriptors['fr_alkyl_halide'],
                    descriptors['fr_allylic_oxid'],
                    descriptors['fr_amide'],
                    descriptors['fr_amidine'],
                    descriptors['fr_aniline'],
                    descriptors['fr_aryl_methyl'],
                    descriptors['fr_azide'],
                    descriptors['fr_azo'],
                    descriptors['fr_barbitur'],
                    descriptors['fr_benzene'],
                    descriptors['fr_benzodiazepine'],
                    descriptors['fr_bicyclic'],
                    descriptors['fr_diazo'],
                    descriptors['fr_dihydropyridine'],
                    descriptors['fr_epoxide'],
                    descriptors['fr_ester'],
                    descriptors['fr_ether'],
                    descriptors['fr_furan'],
                    descriptors['fr_guanido'],
                    descriptors['fr_halogen'],
                    descriptors['fr_hdrzine'],
                    descriptors['fr_hdrzone'],
                    descriptors['fr_imidazole'],
                    descriptors['fr_imide'],
                    descriptors['fr_isocyan'],
                    descriptors['fr_isothiocyan'],
                    descriptors['fr_ketone'],
                    descriptors['fr_ketone_Topliss'],
                    descriptors['fr_lactam'],
                    descriptors['fr_lactone'],
                    descriptors['fr_methoxy'],
                    descriptors['fr_morpholine'],
                    descriptors['fr_nitrile'],
                    descriptors['fr_nitro'],
                    descriptors['fr_nitro_arom'],
                    descriptors['fr_nitro_arom_nonortho'],
                    descriptors['fr_nitroso'],
                    descriptors['fr_oxazole'],
                    descriptors['fr_oxime'],
                    descriptors['fr_para_hydroxylation'],
                    descriptors['fr_phenol'],
                    descriptors['fr_phenol_noOrthoHbond'],
                    descriptors['fr_phos_acid'],
                    descriptors['fr_phos_ester'],
                    descriptors['fr_piperdine'],
                    descriptors['fr_piperzine'],
                    descriptors['fr_priamide'],
                    descriptors['fr_prisulfonamd'],
                    descriptors['fr_pyridine'],
                    descriptors['fr_quatN'],
                    descriptors['fr_sulfide'],
                    descriptors['fr_sulfonamd'],
                    descriptors['fr_sulfone'],
                    descriptors['fr_term_acetylene'],
                    descriptors['fr_tetrazole'],
                    descriptors['fr_thiazole'],
                    descriptors['fr_thiocyan'],
                    descriptors['fr_thiophene'],
                    descriptors['fr_unbrch_alkane'],
                    descriptors['fr_urea'],

                ]])

                # Make prediction
                prediction = model.predict(model_input)
                if prediction == 0:
                    permeability = "impermeable"
                elif prediction == 1:
                    permeability = "moderate"
                else:
                    permeability = "good"

                st.write("### Prediction")
                st.write(f"The predicted permeability is: {permeability}")

                # Display descriptors
                st.write("### Molecular Descriptors")
                st.dataframe(df)
            else:
                st.error("Invalid SMILES string. Please enter a valid SMILES.")



        except Exception as e:
            st.error(f"An error occurred: {e}")

# Description page
elif page == "Description":
    st.title("Description")
    st.write("### Model Details")
    st.write("This section provides details about the machine learning model used.")



    # Model performance metrics
    performance_metrics = {
        "Metric": ["Accuracy", "F1 Score"],
        "Score": [0.896, 0.890]
    }
    performance_df = pd.DataFrame(performance_metrics)
    st.write("#### Model Performance")
    st.table(performance_df)

    # Permeability criteria
    permeability_criteria = {
        "Permeability range": ["Good", "Moderate/Low", "Impermeable"],
        "Criteria": ["LogP >= -5", "-7 < LogP < -5", "LogP < -7"]
    }
    permeability_df = pd.DataFrame(permeability_criteria)
    st.write("#### Permeability Criteria")
    st.table(permeability_df)

    st.title("Database")
    st.write("This section includes information about the molecular database.")
    st.write("CycPeptMPDB (Cyclic Peptide Membrane Permeability Database) is the largest web-accessible database of membrane permeability of cyclic peptide. The latest version provides the information for 7,334 structurally diverse cyclic peptides collected from 47 publications. These cyclic peptides are composed of 312 types Monomers (substructures). ")
    st.write("### Source of Training Dataset")
    st.write("[Cyclic Peptide Database](http://cycpeptmpdb.com/)")

    df = pd.read_csv("CycPeptMPDB_Peptide.csv")
    st.write(df.head())


# Contact page
elif page == "Contact":
    st.title("Contact")
    st.write("You can reach me at:")
    st.write("Email: Hj728490@gmail.com")
    st.write("Mobile: +91 9024990040")
    st.write("We are open to any Feedbacks, suggestions and contributions",style = "center")
