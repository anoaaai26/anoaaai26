import pandas as pd
from pathlib import Path
from sas7bdat import SAS7BDAT
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

###################################################################################
# Respiratory (from https://physionet.org/content/mimic2-iaccd/1.0/)

respiratory_feature_types = {
	"aline_flg": "discrete",
	"icu_los_day": "continuous",
	"hospital_los_day": "continuous",
	"age": "continuous",
	"gender_num": "discrete",
	"weight_first": "continuous",
	"bmi": "continuous",
	"sapsi_first": "continuous",
	"sofa_first": "continuous",
	"service_unit": "discrete",
	"service_num": "discrete",
	"day_icu_intime": "discrete",
	"day_icu_intime_num": "discrete",
	"hour_icu_intime": "discrete",
	"hosp_exp_flg": "discrete",
	"icu_exp_flg": "discrete",
	"mort_day_censored": "continuous",
	"censor_flg": "discrete",
	"sepsis_flg": "discrete",
	"chf_flg": "discrete",
	"afib_flg": "discrete",
	"renal_flg": "discrete",
	"liver_flg": "discrete",
	"copd_flg": "discrete",
	"cad_flg": "discrete",
	"stroke_flg": "discrete",
	"mal_flg": "discrete",
	"resp_flg": "discrete",
	"map_1st": "continuous",
	"hr_1st": "continuous",
	"temp_1st": "continuous",
	"spo2_1st": "continuous",
	"abg_count": "discrete",
	"wbc_first": "continuous",
	"hgb_first": "continuous",
	"platelet_first": "continuous",
	"sodium_first": "continuous",
	"potassium_first": "continuous",
	"tco2_first": "continuous",
	"chloride_first": "continuous",
	"bun_first": "continuous",
	"creatinine_first": "continuous",
	"po2_first": "continuous",
	"pco2_first": "continuous",
	"iv_day_1": "continuous",
	"class": "discrete",
}
df = pd.read_csv("dataset/respiratory/full_cohort_data.csv")
df = df.rename(columns={"day_28_flg": "class"})
target = df.pop("class")
df["class"] = target
continuous_columns = [
    column for column, feature_type in respiratory_feature_types.items()
    if feature_type == "continuous" and column in df.columns
]
for column in continuous_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
if continuous_columns:
    medians = df[continuous_columns].median()
    df[continuous_columns] = df[continuous_columns].fillna(medians)
df.dropna(inplace=True)

df.to_csv("dataset/respiratory/respiratory.csv", index=False)
info_path = Path("dataset/respiratory/respiratory.info")
info_path.parent.mkdir(parents=True, exist_ok=True)
lines = []
for index, column in enumerate(df.columns[:-1], start=1):
    feature_type = respiratory_feature_types[column]
    lines.append(f"{index} {feature_type}")
lines.append(f"class {respiratory_feature_types['class']}")
lines.append("LABEL_POS -1")
info_path.write_text("\n".join(lines) + "\n")

####################################################################################
# Lung Cancer (from https://clinicaltrials.gov/ct2/show/NCT00003299)

with SAS7BDAT('dataset/lungcancer/c9732_demographic.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df = df.assign(chemo_cycle_group=pd.cut(df['CHEMO_CYCLE'], bins=[0, 5, 10, 999], labels=['1','2','3',]))
df['STATUS'].replace({3:1}, inplace=True)
df['STATUS'].replace({1:0, 2:1}, inplace=True)
columns = [
    'GENDER',
    'AGE',
    'RACE',
    'PS',
    'NUM_META',
    'chemo_cycle_group',
    'TRT_ARM',
    'STATUS',
]
rename_dict = {
    'GENDER': 'gender',
    'AGE': 'age',
    'RACE': 'race',
    'PS': 'ECOG performance status',
    'NUM_META': 'num_metastatic',
    'chemo_cycle_group': 'chemotherapy cycle group',
    'TRT_ARM':'treatment arm',
    'STATUS': 'class',
}
####################################################################################
# Breast Cancer (from https://clinicaltrials.gov/ct2/show/NCT00041119)

with SAS7BDAT('dataset/breastcancer/all_finalb.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
columns = [
    'RACE_ID',
    'stra1',
    'stra3',
    'indrx',
    'OH002',
    'OH003',
    'OH004',
    'OH005',
    'OH011',
    'OH016',
    'OH027',
    'OH036',
    'OH037',
    'num_pos_nodes',
    'tsize',
    'survstat', # 0: alive, 1: dead
]
rename_dict = {
    'RACE_ID': 'race',
    'stra1': 'post-menopause',
    'stra3': 'human epidermal growth factor receptor 2 is positive',
    'indrx': 'treatment',
    'OH002': 'tumor laterality',
    'OH003': 'estrogen receptor positive',
    'OH004': 'progesterone receptor positive',
    'OH005': 'cancer histologic grade',
    'OH011': 'prior hormonal therapy',
    'OH016': 'prior chemotherapy',
    'OH027': 'biopsy type',
    'OH036': 'sentinel node biospy',
    'OH037': 'axillary dissection',
    'num_pos_nodes': 'number of positive axillary nodes',
    'tsize': 'tumor size',
    'survstat': 'class',
}
df_processed = df[columns].rename(columns=rename_dict)
num_feat_list = ['number of positive axillary nodes', 'tumor size']
df_processed['number of positive axillary nodes'] = (
    df_processed['number of positive axillary nodes']
    .astype(str)
    .str.strip()
    .replace({'>3': '4', 'nan': np.nan})
)
df_processed[num_feat_list] = df_processed[num_feat_list].apply(pd.to_numeric, errors='coerce')
df_processed['number of positive axillary nodes'] = df_processed['number of positive axillary nodes'].fillna(
    df_processed['number of positive axillary nodes'].median()
)
df_processed['tumor laterality'] = pd.to_numeric(df_processed['tumor laterality'], errors='coerce')
if df_processed['tumor laterality'].isna().any():
    df_processed['tumor laterality'] = df_processed['tumor laterality'].fillna(
        df_processed['tumor laterality'].mode().iloc[0]
    )
df_processed['tumor laterality'] = df_processed['tumor laterality'].astype(int)
df_processed.fillna(method='ffill', inplace=True)
df_processed.to_csv('dataset/breastcancer/breastcancer.csv', index=False)

####################################################################################
# Colon Cancer (from https://clinicaltrials.gov/ct2/show/NCT00079274)

df = pd.read_csv("dataset/coloncancer/characteristic.csv")
bin_feat_list = []
num_feat_list = []
df = df.rename(columns={
    'ADHERENC': 'adherence',
    'agecat': 'age',
    'ARM': 'arms',
    'BAD_TOX': 'serious adverse effect',
    'BWL_OBS': 'bowel obstruction',
    'BWL_PERF': 'bowel perforation',
    'HISTO_G': 'histology',
    'PS': 'ECOG performance score',
    'racecat': 'race',
    'SEX':'sex',
    'wild':'biomarker KRAS',
    'bmi2':'bmi',
}
)
df['age'] = df['age'].replace({
    '< 40': 40,
    '40-69': 55,
    '>= 70': 70,
}).astype(int)
df['ECOG performance score'] = df['ECOG performance score'].fillna(0).astype(int)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df_processed = df[[
    'adherence',
    'age',
    'arms',
    'serious adverse effect',
    'bowel obstruction',
    'bowel perforation',
    'histology',
    'ECOG performance score',
    'race',
    'sex',
    'biomarker KRAS',
    'bmi',
    'mask_id',
]]
df_obj = pd.read_csv("dataset/coloncancer/objectives.csv")
df_obj['target_label'] = df_obj['fustat8']
df_tox = pd.read_csv('dataset/coloncancer/tox.csv')
df_tox['is serious'] = df_tox['GRADE'].apply(lambda x: 1 if x > 3 else 0)
ae_list = df_tox['tox'].value_counts().index.tolist()
ae_name_list = []
for ae in ae_list:
    ae_name = 'adverse effect: ' + ae.lower()
    df_tox[ae_name] = np.zeros(len(df_tox))
    df_tox.loc[df_tox['tox']==ae,ae_name] = 1
    df_tox[ae_name] = df_tox[ae_name] * df_tox['is serious']
    ae_name_list.append(ae_name)
df_tox = df_tox[ae_name_list+['mask_id']].groupby('mask_id').max().reset_index()
sub_ae_cols = ['adverse effect: thrombosis', 'adverse effect: hypersensitivity', 'adverse effect: infarction', 'adverse effect: diarrhea']
df_processed = df_processed.merge(df_tox[sub_ae_cols+['mask_id']],how='left').fillna(0)
df_processed = df_processed.merge(df_obj[['target_label', 'mask_id']])
df_processed = df_processed.drop('mask_id', axis=1)
df_processed = df_processed.rename(columns={'target_label': 'class'})
ordered_columns = [col for col in df_processed.columns if col != 'class'] + ['class']
df_processed = df_processed[ordered_columns]
df_processed.to_csv('dataset/coloncancer/coloncancer.csv', index=False)

####################################################################################
# Breast Cancer 2 (from https://clinicaltrials.gov/ct2/show/NCT00312208)

num_feat_list = []
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/demog.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df_processed = df[['AGENO','RUSUBJID']].rename(columns={'AGENO':'age'})
num_feat_list.append('age')
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/ae.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df = df[df['SERIOUS'] == 'Yes']
ae_list = df['AE_SEL'].value_counts()[1:6].index.tolist()
ae_cols = []
for ae in ae_list:
    ae_name = 'adverse effect: ' + ae.lower()
    df[ae_name] = np.zeros(len(df))
    df.loc[df['AE_SEL']==ae,ae_name] = 1
    ae_cols.append(ae_name)
df_ae = df[ae_cols+['RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df_ae.replace({1:'Yes',0:'No'},inplace=True)
df_processed = df_processed.merge(df_ae,on='RUSUBJID',how='left')
df_processed[ae_cols] = df_processed[ae_cols].fillna('No')
df_processed[num_feat_list]=df_processed[num_feat_list].fillna(df_processed[num_feat_list].median())
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/antitumo.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
tumo_med_list = df['CTX_L'].value_counts()[:5].index.tolist()
tumo_med_cols = []
for med in tumo_med_list:
    med_name = 'anti-tumor therapy: ' + med.lower()
    df[med_name] = np.zeros(len(df))
    df.loc[df['CTX_L']==med, med_name] = 1
    tumo_med_cols.append(med_name)
df_tumomed = df[tumo_med_cols+['RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df_tumomed.replace({1:'Yes',0:'No'},inplace=True)
df_processed = df_processed.merge(df_tumomed,on='RUSUBJID',how='left')
df_processed[tumo_med_cols] = df_processed[tumo_med_cols].fillna('No')
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/diag2.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df = df[['RESALN','POSALN','RUSUBJID']].rename(columns={'RESALN':'number of resected axillary node', 'POSALN':'numer of positive axillary node'})
num_feat_list.append('number of resected axillary node')
num_feat_list.append('numer of positive axillary node')
df_processed = df_processed.merge(df, on='RUSUBJID', how='left')
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/death.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df['class'] = np.ones(len(df))
df_processed = df_processed.merge(df[['class','RUSUBJID']],on='RUSUBJID',how='left')
df_processed['class'] = df_processed['class'].fillna(0)
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/diag3.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df = df.rename(columns={'PTSIZE':'primary tumor size'})
df_processed = df_processed.merge(df[['primary tumor size', 'RUSUBJID']],on='RUSUBJID',how='left')
num_feat_list.append('primary tumor size')
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/hormrec.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df['estrogen receptor positive'] = df['ERSTA'].apply(lambda x: 1 if x == 'Positive' else 0)
df['progesterone receptor positive'] = df['PGRSTA'].apply(lambda x: 1 if x == 'Positive' else 0)
df = df[['estrogen receptor positive','progesterone receptor positive','RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df[['estrogen receptor positive','progesterone receptor positive']] = df[['estrogen receptor positive','progesterone receptor positive']].replace({1:'Yes',0:'No'})
df_processed = df_processed.merge(df,on='RUSUBJID',how='left')
with SAS7BDAT('dataset/breastcancer2/XRP6976_TAX_GMA_Data/vital.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df = df.rename(columns={'HT':'height','WT': 'weight'})
df = df[['weight','height','RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df['weight'] = df['weight'].replace({'A':np.nan,'> 125':np.nan, 'I':np.nan, '> 275':np.nan})
df[['weight','height']] = df[['weight','height']].apply(pd.to_numeric, errors='coerce')
df[['weight','height']] = df[['weight','height']].fillna(df[['weight','height']].median())
df_processed = df_processed.merge(df, on='RUSUBJID',how='left')
num_feat_list.append('weight')
num_feat_list.append('height')
df_processed[num_feat_list] = df_processed[num_feat_list].fillna(df_processed[num_feat_list].median())
output_df = df_processed.drop('RUSUBJID', axis=1)
ordered_columns = [col for col in output_df.columns if col != 'class'] + ['class']
output_df = output_df[ordered_columns]
output_df.to_csv('dataset/breastcancer2/breastcancer2.csv', index=False)

####################################################################################
# Chemotherapy (from https://clinicaltrials.gov/ct2/show/NCT00694382)

with SAS7BDAT('dataset/chemotherapy/AVE5026_EFC6521_data/ae.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df['AESEV'].value_counts()
df['adverse effect'] = df['AESEV'].apply(lambda x: x.lower()) + ' ' + df['AEDECOD'].apply(lambda x: x.lower())
ae_list = df['adverse effect'].apply(lambda x: x if 'severe'in x else '').value_counts()[1:11].index.tolist()
ae_cols = []
for ae in ae_list:
    ae_name = 'aderse effect: ' + ae
    df[ae_name] = np.zeros(len(df))
    df.loc[df['adverse effect']==ae,ae_name] = 1
    ae_cols.append(ae_name)
df_processed = df[ae_cols+['RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df_processed=df_processed.replace({0:'Yes',1:'No'})
with SAS7BDAT('dataset/chemotherapy/AVE5026_EFC6521_data/cm.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
drug_list = df['CMDECOD'].value_counts()[:10].index.tolist()
drug_list = [d for d in drug_list if len(d)>0]
drug_name_list = []
drug_list
for drug in drug_list:
    drug_name = 'medication: ' + drug.lower()
    drug_name_list.append(drug_name)
    df[drug_name] = np.zeros(len(df))
    df.loc[df['CMDECOD'] == drug, drug_name] = 1
df_drug = df[drug_name_list+['RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df_drug=df_drug.replace({0:'Yes',1:'No'})
df_processed = df_drug.merge(df_processed, on='RUSUBJID', how='left')
df_processed.fillna('No', inplace=True)
with SAS7BDAT('dataset/chemotherapy/AVE5026_EFC6521_data/dm.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df_dm = df[['AGEC','SEX','RUSUBJID']]
df_dm = df_dm.rename(columns={'AGEC':'age','SEX':'sex'})
df_processed = df_processed.merge(df_dm, on='RUSUBJID', how='left')
with SAS7BDAT('dataset/chemotherapy/AVE5026_EFC6521_data/mh.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
dz_list = df['MHDECOD'].value_counts()[:10].index.tolist()
df['mh_status'] = df['MHOCCUR'].apply(lambda x: 0 if x == 'N' else 1)
dz_name_list = []
for dz in dz_list:
    dz_name = 'historical disease: ' + dz.lower()
    dz_name_list.append(dz_name)
    df[dz_name] = (df['MHDECOD'] == dz) * df['mh_status']
df_dz = df[dz_name_list+['RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df_dz = df_dz.replace({0:'No',1:'Yes'})
df_processed = df_dz.merge(df_processed, on='RUSUBJID', how='left')
df_processed['age'].replace({'>85':'85'},inplace=True)
df_processed['age'] = pd.to_numeric(df_processed['age'], errors='coerce')
df_processed['age'].fillna(df_processed['age'].median(),inplace=True)
df_processed['age'] = df_processed['age'].astype(int)
df_processed.fillna('No',inplace=True)
with SAS7BDAT('dataset/chemotherapy/AVE5026_EFC6521_data/ds.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df['target_label'] = df['DSSCAT'].apply(lambda x:1 if x == 'DEATH' else 0)
df_label = df[['target_label','RUSUBJID']].groupby('RUSUBJID').max().reset_index()
df_processed = df_processed.merge(df_label, on='RUSUBJID')
with SAS7BDAT('dataset/chemotherapy/AVE5026_EFC6521_data/lb.sas7bdat', skip_header=False) as reader:
    df = reader.to_data_frame()
df = df[df['LBBLFL'] == 'Y']
lb_list = df['LBTEST'].value_counts()[:10].index.tolist()
lb_name_list = []
for lb in lb_list:
    lb_name = 'lab test: ' + lb.lower()
    lb_name_list.append(lb_name)
    df_lb = df[df['LBTEST'] == lb][['LBSTRESN','RUSUBJID']].groupby('RUSUBJID').mean().reset_index()
    df_lb = df_lb.rename(columns = {'LBSTRESN':lb_name})
    df_processed = df_processed.merge(df_lb, on='RUSUBJID', how='left')
df_lb = df[df['LBTEST'] == lb][['LBSTRESN','RUSUBJID']].groupby('RUSUBJID').mean().reset_index()
numerical_features = ['age'] + lb_name_list
numerical_features = [col for col in numerical_features if col in df_processed.columns]
if numerical_features:
    medians = df_processed[numerical_features].median()
    df_processed[numerical_features] = df_processed[numerical_features].fillna(medians)
df_processed = df_processed.rename(columns={'RUSUBJID':'patient_id', 'target_label':'class'})
df_processed = df_processed.drop('patient_id', axis=1)
ordered_columns = [col for col in df_processed.columns if col != 'class'] + ['class']
df_processed = df_processed[ordered_columns]
df_processed.to_csv('dataset/chemotherapy/chemotherapy.csv', index=False)

####################################################################################
# Bone Marrrow (from https://clinicaltrials.gov/ct2/show/NCT03041311)

# with SAS7BDAT('dataset/bonemarrow/adsl.sas7bdat', skip_header=False) as reader:
#     df = reader.to_data_frame()
# df['age'] = df['AGE'].apply(lambda x: x.split('-')[0]).replace({'>=80':'80'}).astype(int)
# df['ECOG performance score'] = df['ECOGSCR']
# df['height'] = df['HTCM']
# df['weight'] = df['WTKG']
# df['target_label'] = df['DCSREAS'].apply(lambda x: 1 if x =='DEATH' else 0)
# df_processed = df[['USUBJID','age','ECOG performance score','SEX','height','weight','target_label']]
# with SAS7BDAT('dataset/bonemarrow/adcm.sas7bdat', skip_header=False) as reader:
#     df = reader.to_data_frame()
# drug_list = [d for d in df['CMDECOD'].value_counts()[:10].index.tolist() if len(d) > 0]
# # df['baseline_status'] = df['CMPRIOR'].apply(lambda x: 1 if x == 'Yes' else 0)
# for drug in drug_list:
#     df[drug] = np.zeros(len(df))
#     # df.loc[(df['CMDECOD']==drug) & (df['baseline_status']==1), drug] = 1
#     df.loc[(df['CMDECOD']==drug), drug] = 1
# df_drug = df[drug_list+['USUBJID']].groupby('USUBJID').max().reset_index()
# df_processed = df_processed.merge(df_drug,on='USUBJID')
# with SAS7BDAT('dataset/bonemarrow/admh.sas7bdat', skip_header=False) as reader:
#     df = reader.to_data_frame()
# dz_list=[dz for dz in df['MHDECOD'].value_counts()[:10].index.tolist() if len(dz)>0]
# for dz in dz_list:
#     df[dz]=np.zeros(len(df))
#     df.loc[df['MHDECOD']==dz,dz]=1
# df_dz=df[dz_list+['USUBJID']].groupby('USUBJID').max().reset_index()
# df_processed=df_processed.merge(df_dz,on='USUBJID',how='outer').fillna(0)
# df_processed = df_processed.rename(columns={'USUBJID':'patient_id', 'target_label':'class'})
# df_processed = df_processed.drop('patient_id', axis=1)
# df_processed['SEX'] = df_processed['SEX'].apply(lambda x: 0 if x == 'M' else 1)
# ordered_columns = [col for col in df_processed.columns if col != 'class'] + ['class']
# df_processed = df_processed[ordered_columns]
# df_processed.to_csv('dataset/bonemarrow/bonemarrow.csv', index=False)