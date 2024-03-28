import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main() :
    toolbox = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/neurocognition/nc_y_nihtb.csv", dtype=str)
    toolbox_filtered = toolbox[toolbox['eventname']=='baseline_year_1_arm_1']
    toolbox_scores = toolbox.filter(like='uncorrected')
    scores_list = toolbox_scores.columns
    temp = scores_list.drop('nihtbx_fluidcomp_uncorrected')
    temptwo = temp.drop('nihtbx_cryst_uncorrected')
    scores_labels = temptwo.drop('nihtbx_totalcomp_uncorrected')
    nih_scores = toolbox_filtered[scores_labels]
    nihscores = pd.merge(toolbox_filtered['src_subject_id'], toolbox_filtered[scores_labels], left_index=True, right_index=True)

    cash_choice = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/neurocognition/nc_y_cct.csv", dtype=str)
    cash_scores = cash_choice[['src_subject_id', 'cash_choice_task']]
    cash_scores['cash_choice_task'] = cash_scores['cash_choice_task'].replace({'3': 'NaN'})      
    scores = pd.merge(nihscores, cash_scores, on='src_subject_id', how='outer')

    little_man = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/neurocognition/nc_y_lmt.csv", dtype=str)
    little_man['lmt_scr_efficiency'][1688] = 'NaN'
    little_man['lmt_scr_efficiency'][4553] = 'NaN'
    little_man['lmt_scr_efficiency'][12962] = 'NaN'
    little_man['lmt_scr_efficiency'][16761] = 'NaN'
    little_man['lmt_scr_efficiency'][21062] = 'NaN'
    little_man['lmt_scr_efficiency'][21647] = 'NaN'
    little_filtered = little_man[little_man['eventname']=='baseline_year_1_arm_1']
    lmt_score = little_filtered[['src_subject_id', 'lmt_scr_perc_correct']]
    scores = pd.merge(scores, lmt_score, on='src_subject_id', how='outer')

    ravlt = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/neurocognition/nc_y_ravlt.csv", dtype=str)
    ravlt_filtered = ravlt[ravlt['eventname']=='baseline_year_1_arm_1']
    ravlt_scores = ravlt_filtered[['src_subject_id', 'pea_ravlt_sd_listb_tc']]
    scores = pd.merge(scores, ravlt_scores, on='src_subject_id', how='outer')

    wisc = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/neurocognition/nc_y_wisc.csv", dtype=str)
    wisc_scores = wisc[['src_subject_id', 'pea_wiscv_trs']]
    scores = pd.merge(scores, wisc_scores, on='src_subject_id', how='outer')

    cbcl = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/mental-health/mh_p_cbcl.csv", dtype=str)
    cbcl_filtered = cbcl[cbcl['eventname']=='baseline_year_1_arm_1']
    cbcl_scores = cbcl_filtered[['src_subject_id', 'cbcl_scr_dsm5_anxdisord_r', 'cbcl_scr_dsm5_depress_r', 'cbcl_scr_dsm5_adhd_r',
                                'cbcl_scr_dsm5_conduct_r', 'cbcl_scr_dsm5_opposit_r', 'cbcl_scr_dsm5_somaticpr_r', 'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_somatic_r',
                                'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_withdep_r', 'cbcl_scr_syn_aggressive_r', 'cbcl_scr_syn_rulebreak_r',
                                'cbcl_scr_syn_anxdep_r', 'cbcl_scr_syn_social_r']]
    scores = pd.merge(scores, cbcl_scores, on='src_subject_id', how='outer')

    prosocial = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/culture-environment/ce_p_psb.csv", dtype=str)
    int_series = prosocial['psb_p_ss_mean'].astype(float)
    prosocial['psb_p_ss'] = [item * 3 for item in int_series]
    prosocial['psb_p_ss'] = prosocial['psb_p_ss'].astype(str)
    prosocial_filtered =  prosocial[prosocial['eventname']=='baseline_year_1_arm_1']
    prosocial_scores = prosocial_filtered[['src_subject_id', 'psb_p_ss']]
    scores = pd.merge(scores, prosocial_scores, on='src_subject_id', how='outer')

    srs = pd.read_csv("/data/NIMH_scratch/zwallymi/tabulated_release5/core/mental-health/mh_p_ssrs.csv")
    srs_filtered = srs[srs['eventname']=='1_year_follow_up_y_arm_1']
    srs_scores = srs_filtered[['src_subject_id', 'ssrs_p_ss_sum']]
    scores = pd.merge(scores, srs_scores, on='src_subject_id', how='outer')

    mean = {}
    std = {}
    for i in range(1, 27) :
        scores[scores.columns[i]] = pd.to_numeric(scores[scores.columns[i]], errors='coerce')
        mean[scores.columns[i]] = np.mean(scores[scores.columns[i]])
        std[scores.columns[i]] = np.std(scores[scores.columns[i]])

if __name__ == "__main__" :
    main()