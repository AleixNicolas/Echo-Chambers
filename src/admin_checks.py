import pandas as pd
import os
import config

def process_admin(raw_csv_path):
    """Processes demographic checks and tracks participant attrition."""
    df_p2 = pd.read_csv(raw_csv_path)
    df_p2['prolific_id'] = df_p2.get('participant.label', df_p2.get('participant.code'))

    # Eligibility & Attrition
    participation_cols = [f'phase_2.{i}.player.participated_this_round' for i in range(1, config.ROUNDS + 1)]
    avail_cols = [c for c in participation_cols if c in df_p2.columns]
    
    df_p2['total_completed'] = df_p2[avail_cols].fillna(0).sum(axis=1)
    
    final_round_col = f'phase_2.{config.ROUNDS}.player.participated_this_round'
    df_p2['bonus_eligible'] = (df_p2['total_completed'] >= (config.ROUNDS - 1)) & (df_p2.get(final_round_col, 0) == 1)
    df_p2['lottery_eligible'] = (df_p2['total_completed'] == config.ROUNDS)
    df_p2['dropped_out'] = (df_p2['total_completed'] < config.ROUNDS).astype(int)

    eligibility_df = df_p2[['prolific_id', 'total_completed', 'bonus_eligible', 'lottery_eligible']].sort_values('prolific_id')
    eligibility_df.to_csv(os.path.join(config.EMPIRICAL_PROCESSED_DIR, 'eligibility_results.csv'), index=False)

    # Text Summary Output
    stats_dir = os.path.join(config.EMPIRICAL_RESULTS_DIR, 'stats_summaries')
    os.makedirs(stats_dir, exist_ok=True)
    
    lines = ["==========================================", f"  ADMINISTRATIVE SUMMARY - {config.CURRENT_EMPIRICAL_TRIAL_ID}", "==========================================\n"]
    lines.append(f"Total Participants Enrolled: {len(df_p2)}")
    
    for r in range(1, config.ROUNDS + 1):
        col = f'phase_2.{r}.player.participated_this_round'
        if col in df_p2.columns: lines.append(f"  Completed Round {r}: {int(df_p2[col].fillna(0).sum())}")
            
    lines.append(f"\nTotal Dropped Out (Incomplete): {df_p2['dropped_out'].sum()}")
    lines.append(f"Eligible for Bonus Payout: {df_p2['bonus_eligible'].sum()}")

    with open(os.path.join(config.EMPIRICAL_RESULTS_DIR, 'completion_summary.txt'), 'w') as f: 
        f.write("\n".join(lines))