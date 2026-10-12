import pandas as pd
from pathlib import Path
import config

def process_admin(raw_csv_path, global_exclusions=None, out_dir=None):
    df_p2 = pd.read_csv(raw_csv_path, low_memory=False)
    
    df_p2['prolific_id'] = df_p2.get('participant.label', df_p2.get('participant.code', 'UNKNOWN_ID'))

    if global_exclusions:
        clean_globals = [str(x).strip().lower() for x in global_exclusions]
        df_p2 = df_p2[~df_p2['prolific_id'].astype(str).str.lower().isin(clean_globals)]

    node_col_base = f'phase_2.1.player.{config.EMPIRICAL_BASE_TOPIC}_node_id'
    node_col_gen = 'phase_2.1.player.node_id'
    
    if node_col_base in df_p2.columns:
        valid_nodes = df_p2[node_col_base].notna() & (df_p2[node_col_base].astype(str).str.strip() != "-1") & (df_p2[node_col_base].astype(str).str.strip() != "")
    elif node_col_gen in df_p2.columns:
        valid_nodes = df_p2[node_col_gen].notna() & (df_p2[node_col_gen].astype(str).str.strip() != "-1") & (df_p2[node_col_gen].astype(str).str.strip() != "")
    else:
        valid_nodes = pd.Series(True, index=df_p2.index)
        
    df_p2 = df_p2[valid_nodes].copy()

    participation_cols = [f'phase_2.{i}.player.participated_this_round' for i in range(1, config.ROUNDS + 1)]
    avail_cols = [c for c in participation_cols if c in df_p2.columns]
    
    for c in avail_cols:
        df_p2[c] = pd.to_numeric(df_p2[c], errors='coerce').fillna(0)
        
    df_p2['total_completed'] = df_p2[avail_cols].sum(axis=1)
    
    final_round_col = f'phase_2.{config.ROUNDS}.player.participated_this_round'
    final_round_status = df_p2[final_round_col] if final_round_col in df_p2.columns else 0
        
    df_p2['bonus_eligible'] = (df_p2['total_completed'] >= (config.ROUNDS - 1)) & (final_round_status == 1)
    df_p2['lottery_eligible'] = (df_p2['total_completed'] == config.ROUNDS) & (final_round_status == 1)
    df_p2['dropped_out'] = (df_p2['total_completed'] < config.ROUNDS).astype(int)

    def get_payout_status(row):
        if row['lottery_eligible']: return "Bonus + Lottery"
        elif row['bonus_eligible']: return "Bonus Only"
        else: return "No Payout"

    df_p2['payout_status'] = df_p2.apply(get_payout_status, axis=1)

    eligibility_df = df_p2[['prolific_id', 'total_completed', 'payout_status', 'bonus_eligible', 'lottery_eligible']].sort_values('prolific_id')
    eligibility_df.to_csv(config.EMPIRICAL_PROCESSED_DIR / 'eligibility_results.csv', index=False)

    lines = ["==========================================", f"  ADMINISTRATIVE SUMMARY - {config.CURRENT_EMPIRICAL_TRIAL_ID}", "==========================================\n"]
    lines.append(f"Total Actual In-Network Participants: {len(df_p2)}")
    lines.append(f"(Overflow/waiting room users ignored)\n")
    
    for r in range(1, config.ROUNDS + 1):
        col = f'phase_2.{r}.player.participated_this_round'
        if col in df_p2.columns: 
            lines.append(f"  Completed Round {r}: {int(df_p2[col].sum())}")
            
    true_dropouts = (df_p2['total_completed'] < (config.ROUNDS - 1)).sum()
    lines.append(f"\nTotal Dropped Out (Missed >=2 Rounds): {true_dropouts}")
    
    lines.append("\n--- PAYOUT BREAKDOWN ---")
    earned_both = (df_p2['payout_status'] == 'Bonus + Lottery').sum()
    earned_bonus = (df_p2['payout_status'] == 'Bonus Only').sum()
    earned_nothing = (df_p2['payout_status'] == 'No Payout').sum()
    
    lines.append(f"Earned Bonus + Lottery (Perfect Attendance) : {earned_both}")
    lines.append(f"Earned Bonus Only (Missed 1 round, did R5)  : {earned_bonus}")
    lines.append(f"Earned No Payout (Missed >=2 rounds or R5)  : {earned_nothing}")

    lines.append("\n\n==========================================")
    lines.append(" PROLIFIC ID LOGS (For copy-pasting)")
    lines.append("==========================================")
    
    for status, header in [('Bonus + Lottery', '[ Earned Bonus + Lottery ]'), 
                           ('Bonus Only', '[ Earned Bonus Only ]'), 
                           ('No Payout', '[ Earned No Payout (Review for Rejection/Partial) ]')]:
        lines.append(f"\n{header}")
        ids = df_p2[df_p2['payout_status'] == status]['prolific_id'].dropna().tolist()
        lines.append(", ".join([str(x) for x in ids]) if ids else "None")

    target_dir = Path(out_dir) if out_dir else config.EMPIRICAL_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / 'completion_summary.txt', 'w') as f: 
        f.write("\n".join(lines))