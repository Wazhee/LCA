import numpy as np
import pandas as pd 

src_dir = "results/poisoning_ablation_results.csv"
results_df = pd.read_csv(src_dir)

print("Calculating bias mitigation performance...")

sex = 'M'
for n in [2, 4, 6, 8]:
    no_lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'no_lca')
    ]
    
    lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'lca')
    ]
    
    original_overall_fnr = list(no_lca_df['overall_fnr'].astype(float))
    original_target_fnr = list(no_lca_df['male_fnr'].astype(float))
    
    overall_fnr = list(lca_df['overall_fnr'].astype(float))
    target_fnr = list(lca_df['male_fnr'].astype(float))

    # Compute original average difference (female - male) 
    original_avg_diff = np.mean(np.abs(np.array(original_overall_fnr) - np.array(original_target_fnr)))
    
    # Compute lca average difference (female - male)
    avg_diff = np.mean(np.abs(np.array(overall_fnr) - np.array(target_fnr)))
    
    # Compute percent decrease
    if original_avg_diff != 0:
        percent_decrease = ((original_avg_diff - avg_diff) / original_avg_diff) * 100
    else:
        percent_decrease = 0.0
    print(f"Strength {n}: Original={original_avg_diff:.4f}, LCA={avg_diff:.4f}, Percent decrease={percent_decrease:.2f}%")


print('\n\n')

sex = 'F'
for n in [2, 4, 6, 8]:
    no_lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'no_lca')
    ]
    
    lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'lca')
    ]
    
    original_overall_fnr = list(no_lca_df['overall_fnr'].astype(float))
    original_target_fnr = list(no_lca_df['female_fnr'].astype(float))
    
    overall_fnr = list(lca_df['overall_fnr'].astype(float))
    target_fnr = list(lca_df['female_fnr'].astype(float))

    # Compute original average difference (female - male) 
    original_avg_diff = np.mean(np.abs(np.array(original_overall_fnr) - np.array(original_target_fnr)))
    
    # Compute lca average difference (female - male)
    avg_diff = np.mean(np.abs(np.array(overall_fnr) - np.array(target_fnr)))
    
    # Compute percent decrease
    if original_avg_diff != 0:
        percent_decrease = ((original_avg_diff - avg_diff) / original_avg_diff) * 100
    else:
        percent_decrease = 0.0
    print(f"Strength {n}: Original={original_avg_diff:.4f}, LCA={avg_diff:.4f}, Percent decrease={percent_decrease:.2f}%")



print("\nCalculating disparities between male and female subgroups...")

sex = 'M'
for n in [2, 4, 6, 8]:
    no_lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'no_lca')
    ]
    
    lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'lca')
    ]
    
    original_female_fnr = list(no_lca_df['female_fnr'].astype(float))
    original_male_fnr = list(no_lca_df['male_fnr'].astype(float))
    
    female_fnr = list(lca_df['female_fnr'].astype(float))
    male_fnr = list(lca_df['male_fnr'].astype(float))

    # Compute original average difference (female - male) 
    original_avg_diff = np.mean(np.abs(np.array(original_female_fnr) - np.array(original_male_fnr)))
    
    # Compute lca average difference (female - male)
    avg_diff = np.mean(np.abs(np.array(female_fnr) - np.array(male_fnr)))
    
    # Compute percent decrease
    if original_avg_diff != 0:
        percent_decrease = ((original_avg_diff - avg_diff) / original_avg_diff) * 100
    else:
        percent_decrease = 0.0
    print(f"Strength {n}: Original={original_avg_diff:.4f}, LCA={avg_diff:.4f}, Percent decrease={percent_decrease:.2f}%")


print('\n\n')


sex = 'F'
for n in [2, 4, 6, 8]:
    no_lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'no_lca')
    ]
    
    lca_df = results_df[
        (results_df['strength_level'] == n) &
        (results_df['sex'] == sex) &
        (results_df['apply_lca'] == 'lca')
    ]
    
    original_female_fnr = list(no_lca_df['female_fnr'].astype(float))
    original_male_fnr = list(no_lca_df['male_fnr'].astype(float))
    
    female_fnr = list(lca_df['female_fnr'].astype(float))
    male_fnr = list(lca_df['male_fnr'].astype(float))

    # Compute original average difference (female - male) 
    original_avg_diff = np.mean(np.abs(np.array(original_female_fnr) - np.array(original_male_fnr)))
    
    # Compute lca average difference (female - male)
    avg_diff = np.mean(np.abs(np.array(female_fnr) - np.array(male_fnr)))
    
    # Compute percent decrease
    if original_avg_diff != 0:
        percent_decrease = ((original_avg_diff - avg_diff) / original_avg_diff) * 100
    else:
        percent_decrease = 0.0
    print(f"Strength {n}: Original={original_avg_diff:.4f}, LCA={avg_diff:.4f}, Percent decrease={percent_decrease:.2f}%")