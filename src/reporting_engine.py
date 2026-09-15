import os
import pandas as pd
import numpy as np
import config
from src import data_wrangle, stats_suite

def export_screening_combinations(phase1_csv_path, output_dir):
    if not os.path.exists(phase1_csv_path):
        print(f"[!] Warning: Phase 1 CSV {phase1_csv_path} not found. Skipping screening combinations report.")
        return
    
    df = pd.read_csv(phase1_csv_path, low_memory=False)
    
    def get_screening_opinion(row, topic):
        cols = [
            f'phase_1.1.player.{topic}_opinion_4', 
            f'phase_1.1.player.{topic}_opinion_2',
            f'phase_1.1.player.{topic}_opinion_1',
            f'{topic}_opinion_4', 
            f'{topic}_opinion_2', 
            f'participant.{topic}_opinion_4', 
            f'participant.{topic}_opinion_2'
        ]
        for c in cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip() != '':
                try:
                    val = float(row[c])
                    val = max(1.0, min(5.0, val))
                    if topic in config.INVERT_OPINIONS_FOR:
                        return 6.0 - val
                    return val
                except:
                    pass
        return None 
        
    results = []
    for _, row in df.iterrows():
        c_val = get_screening_opinion(row, config.EMPIRICAL_SIDE_TOPIC)
        i_val = get_screening_opinion(row, config.EMPIRICAL_BASE_TOPIC)
        
        if c_val is not None and i_val is not None:
            results.append({
                'Secondary_Topic': config.get_emp_user_bucket(c_val),
                'Base_Topic': config.get_emp_user_bucket(i_val)
            })
        
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("[!] Warning: No valid screening data found to generate combinations.")
        return
        
    cross_tab = pd.crosstab(res_df['Base_Topic'], res_df['Secondary_Topic'])
    cross_tab = cross_tab.reindex(index=config.USER_CATS, columns=config.USER_CATS, fill_value=0)
    
    lines = ["==========================================", " PHASE 1 SCREENING COMBINATIONS (5x5)", "==========================================\n"]
    lines.append(f"Total Phase 1 Participants Analyzed: {len(res_df)}\n")
    lines.append(f"Rows: Base Topic ({config.EMPIRICAL_BASE_TOPIC}) | Columns: Secondary Topic ({config.EMPIRICAL_SIDE_TOPIC})\n")
    lines.append(cross_tab.to_string())
    
    file_path = os.path.join(output_dir, "Phase1_Screening_Combinations.txt")
    with open(file_path, 'w') as f:
        f.write("\n".join(lines))
    print(f" -> Exported Phase1_Screening_Combinations.txt")


def export_comprehensive_empirical_summary(event_log_df, output_dir, phase2_csv_path, topics=['climate'], global_exclusions=None, round_exclusions=None):
    if event_log_df.empty:
        print("[!] Warning: Event log is empty. Skipping empirical summary report.")
        return

    is_dual = len(topics) > 1
    df2 = pd.read_csv(phase2_csv_path, low_memory=False)
    
    # 1. ALIAS MAPPING (To link participant.code to Prolific IDs for exclusions)
    user_aliases = {}
    for _, row in df2.iterrows():
        aliases = []
        if 'participant.label' in df2.columns and pd.notna(row['participant.label']):
            aliases.append(str(row['participant.label']).strip().lower())
        if 'participant.code' in df2.columns and pd.notna(row['participant.code']):
            aliases.append(str(row['participant.code']).strip().lower())
        for alias in aliases:
            user_aliases[alias] = aliases

    clean_globals = [str(x).strip().lower() for x in global_exclusions] if global_exclusions else []
    
    # 2. STRICT PARTICIPATION FILTERING (Fixed ID matching)
    valid_rows = []
    max_interactive_round = min(config.ROUNDS, 5)
    
    for r in range(1, max_interactive_round + 1):
        part_col = f'phase_2.{r}.player.participated_this_round'
        if part_col in df2.columns and 'participant.code' in df2.columns:
            # Safely check for any truthy value (1, 1.0, True, true)
            is_active = df2[part_col].apply(lambda x: str(x).strip().lower() in ['1', '1.0', 'true', 'yes'])
            active_users = df2[is_active]['participant.code'].dropna().unique()
        else:
            active_users = event_log_df[event_log_df['round'] == r]['user_id'].unique()
            
        cleaned_active_users = []
        clean_round_excl = [str(x).strip().lower() for x in round_exclusions.get(r, [])] if round_exclusions else []
        
        for uid in active_users:
            clean_uid = str(uid).strip().lower()
            my_aliases = user_aliases.get(clean_uid, [clean_uid])
            
            # Scrub out any excluded participants (like the returned study)
            if any(alias in clean_globals for alias in my_aliases) or any(alias in clean_round_excl for alias in my_aliases):
                continue
            else:
                # MUST append as uppercase to match the event_log_df user_ids
                cleaned_active_users.append(str(uid).upper())
                
        for uid in cleaned_active_users:
            valid_rows.append({'user_id': uid, 'round': r})
            
    valid_user_rounds = pd.DataFrame(valid_rows).drop_duplicates()
    
    # Merge will now perfectly align participant codes (e.g. 0GMW279V == 0GMW279V)
    active_event_log_df = pd.merge(event_log_df, valid_user_rounds, on=['user_id', 'round'], how='inner')
    active_event_log_df = active_event_log_df[active_event_log_df['round'] <= 5]

    if is_dual:
        base_t = config.EMPIRICAL_BASE_TOPIC
        side_t = config.EMPIRICAL_SIDE_TOPIC
        
        user_topic_cats = active_event_log_df[['user_id', 'topic', 'user_cat']].drop_duplicates()
        pivot_cats = user_topic_cats.pivot(index='user_id', columns='topic', values='user_cat')
        
        if base_t in pivot_cats.columns and side_t in pivot_cats.columns:
            cross_tab = pd.crosstab(pivot_cats[base_t], pivot_cats[side_t])
            cross_tab = cross_tab.reindex(index=config.USER_CATS, columns=config.USER_CATS, fill_value=0)
            
            cohort_lines = ["==========================================", " ACTUAL PHASE 2 COHORT COMBINATIONS (5x5)", "==========================================\n"]
            cohort_lines.append(f"Total Active Phase 2 In-Network Participants: {len(pivot_cats)}\n")
            cohort_lines.append(f"Rows: Base Topic ({base_t}) | Columns: Secondary Topic ({side_t})\n")
            cohort_lines.append(cross_tab.to_string())
            
            with open(os.path.join(output_dir, "Phase2_Cohort_Combinations.txt"), 'w') as f:
                f.write("\n".join(cohort_lines))
            print(" -> Exported Phase2_Cohort_Combinations.txt")

    def get_2d_profile(row):
        base = 'L' if row.get('chamber') == 'Left' else 'R'
        ucat = row.get('user_cat', 'Center')
        side = 'L' if 'Left' in ucat else ('R' if 'Right' in ucat else 'C')
        return f"{base}{side}"
    
    for topic in topics:
        base_topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        base_topic_df = base_topic_df[base_topic_df['round'] <= 5] 
        
        # Determine the definitive list of real network users (ignoring dropouts/ghosts)
        all_topic_users = base_topic_df['user_id'].unique()
        
        topic_df = active_event_log_df[active_event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else active_event_log_df
        
        lines = ["==========================================", f" EMPIRICAL SUMMARY: {topic.upper()} ", "==========================================\n"]
        
        # 3. ATTRITION BY 2D PROFILE (IN-NETWORK ONLY)
        user_round_counts = valid_user_rounds[valid_user_rounds['user_id'].isin(all_topic_users)].groupby('user_id').size()
        
        # A participant MUST have completed at least 4 rounds (missed <= 1) to be a completer
        completers = user_round_counts[user_round_counts >= (config.ROUNDS - 1)].index.tolist()
        dropouts = list(set(all_topic_users) - set(completers))
        
        lines.append("--- IN-NETWORK ATTRITION BY 2D PROFILE ---")
        lines.append(f"Total In-Network Participants : {len(all_topic_users)}")
        lines.append(f"\nCompleted Valid Session (>= 4 Rounds): {len(completers)}")
        
        comp_df = base_topic_df[base_topic_df['user_id'].isin(completers)].drop_duplicates(subset=['user_id']).copy()
        
        if not comp_df.empty:
            comp_df['2D_Profile'] = comp_df.apply(get_2d_profile, axis=1)
            c_counts = comp_df['2D_Profile'].value_counts()
        else:
            c_counts = {}
            
        for quad in ['LL', 'LC', 'LR', 'RL', 'RC', 'RR']:
            lines.append(f"  {quad.ljust(15)}: {c_counts.get(quad, 0)}")
            
        lines.append(f"\nDropped Out (Missed >= 2 Rounds): {len(dropouts)}")
        if len(dropouts) > 0:
            dropout_df = base_topic_df[base_topic_df['user_id'].isin(dropouts)].drop_duplicates(subset=['user_id']).copy()
            
            if not dropout_df.empty:
                dropout_df['2D_Profile'] = dropout_df.apply(get_2d_profile, axis=1)
                d_counts = dropout_df['2D_Profile'].value_counts()
            else:
                d_counts = {}
                
            for quad in ['LL', 'LC', 'LR', 'RL', 'RC', 'RR']:
                lines.append(f"  {quad.ljust(15)}: {d_counts.get(quad, 0)}")
                
        # 4. ZERO-SHARE TRACKING
        user_action_counts = topic_df.groupby('user_id')['action'].value_counts().unstack(fill_value=0)
        if 'shared' not in user_action_counts.columns: user_action_counts['shared'] = 0
        zero_share_users = user_action_counts[user_action_counts['shared'] == 0].index.tolist()
        
        lines.append(f"\nZero-Share Participants    : {len(zero_share_users)}")
        if len(zero_share_users) > 0:
            for uid in zero_share_users:
                u_row = topic_df[topic_df['user_id'] == uid].iloc[0]
                lines.append(f"  - User ID: {uid} | 2D Profile: {get_2d_profile(u_row)}")
        lines.append("")
        
        # 5. ROUND-BY-ROUND BREAKDOWN BY CHAMBER
        src_metrics = stats_suite.calculate_sourcing_metrics(topic_df)
        def fmt(val): return f"{val:.2f}" if not np.isnan(val) else "N/A "

        for r in range(1, max_interactive_round + 1):
            lines.append(f"\n--- ROUND {r} ---")
            r_df = topic_df[topic_df['round'] == r]
            lines.append(f"  Active Participants: {r_df['user_id'].nunique()}")
            
            if src_metrics and r <= len(src_metrics['rounds']):
                r_idx = r - 1
                lines.append(f"  Sourcing Averages : {fmt(src_metrics['rep_avg'][r_idx])} Repeats | {fmt(src_metrics['pool_avg'][r_idx])} Pool | {fmt(src_metrics['back_avg'][r_idx])} Backlog")

            for chamber in ['Left', 'Right']:
                c_df = r_df[r_df['chamber'] == chamber]
                if c_df.empty: continue
                lines.append(f"\n  [{chamber} Chamber]")
                c_seen = c_df[c_df['action'] == 'seen']
                c_share = c_df[c_df['action'] == 'shared']
                
                for uc in config.USER_CATS:
                    u_seen = c_seen[c_seen['user_cat'] == uc]
                    u_share = c_share[c_share['user_cat'] == uc]
                    tot_cat_seen = len(u_seen)
                    if tot_cat_seen > 0:
                        dist = [len(u_seen[u_seen['item_cat'] == ic]) / tot_cat_seen * 100 for ic in config.ITEM_CATS]
                        rates = [(len(u_share[u_share['item_cat'] == ic]) / len(u_seen[u_seen['item_cat'] == ic]) * 100) if len(u_seen[u_seen['item_cat'] == ic]) > 0 else 0 for ic in config.ITEM_CATS]
                        lines.append(f"    {uc:<12} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                        lines.append(f"    {uc:<12} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")

        # 6. GLOBAL SHARE RATE MATRIX (ALL ROUNDS, BOTH CHAMBERS)
        lines.append("\n" + "="*40)
        lines.append(f" GLOBAL SHARE RATE MATRIX (ALL ROUNDS, BOTH CHAMBERS)")
        lines.append("="*40)
        
        glob_grouped = topic_df.groupby(['user_cat', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
        for col in ['seen', 'shared']:
            if col not in glob_grouped.columns: glob_grouped[col] = 0
                
        glob_grouped['share_rate'] = np.where(glob_grouped['seen'] > 0, glob_grouped['shared'] / glob_grouped['seen'], 0.0)
        glob_matrix = glob_grouped.pivot(index='user_cat', columns='item_cat', values='share_rate').fillna(0)
        glob_matrix = glob_matrix.reindex(index=config.USER_CATS, columns=config.ITEM_CATS).fillna(0)
        lines.append("\n" + glob_matrix.to_string(float_format=lambda x: f"{x:.1%}"))
        lines.append("\n")

        # 7. CHAMBER-ISOLATED SHARE MATRICES
        for chamber in ['Left', 'Right']:
            chamber_df = topic_df[topic_df['chamber'] == chamber]
            if chamber_df.empty: continue
            
            lines.append("\n" + "="*40)
            lines.append(f" {chamber.upper()} CHAMBER: SHARE RATE MATRIX")
            lines.append("="*40)
            
            grouped = chamber_df.groupby(['user_cat', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
            for col in ['seen', 'shared']:
                if col not in grouped.columns: grouped[col] = 0
                    
            grouped['share_rate'] = np.where(grouped['seen'] > 0, grouped['shared'] / grouped['seen'], 0.0)
            
            matrix = grouped.pivot(index='user_cat', columns='item_cat', values='share_rate').fillna(0)
            matrix = matrix.reindex(index=config.USER_CATS, columns=config.ITEM_CATS).fillna(0)
            lines.append("\n" + matrix.to_string(float_format=lambda x: f"{x:.1%}"))
            lines.append("\n")

        safe_suffix = f"_{topic}" if is_dual else ""
        file_path = os.path.join(output_dir, f"Empirical_Comprehensive_Summary{safe_suffix}.txt")
        with open(file_path, 'w') as f: 
            f.write("\n".join(lines))
        print(f" -> Exported full details to Empirical_Comprehensive_Summary{safe_suffix}.txt")


def export_sweep_summary(sweep_aggregator, sweep_dir, k_val, regime_name, priority_str):
    lines = ["==========================================", f" SIMULATION SWEEP SUMMARY", f" Regime: {regime_name} | Priority: {priority_str} | K: {k_val}", "==========================================\n"]
    
    for struct in ['integrated', 'segregated']:
        lines.append(f"=== {struct.upper()} TOPOLOGY ===")
        for n_val in sorted(sweep_aggregator.keys()):
            lines.append(f"\n[ N = {n_val} ]")
            data = sweep_aggregator[n_val][struct]
            
            t_df = data_wrangle.build_simulation_event_log({struct: data})
            
            src_metrics = stats_suite.calculate_sourcing_metrics(t_df)
            def fmt(val): return f"{val:.2f}" if not np.isnan(val) else "N/A "
            
            for rnd in range(1, config.ROUNDS + 1):
                lines.append(f"\n  -- ROUND {rnd} --")
                if src_metrics and rnd <= len(src_metrics['rounds']):
                    r_idx = rnd - 1
                    lines.append(f"       Sourcing Averages : {fmt(src_metrics['rep_avg'][r_idx])} Repeats | {fmt(src_metrics['pool_avg'][r_idx])} Pool | {fmt(src_metrics['back_avg'][r_idx])} Backlog")
                
                rnd_df = t_df[t_df['round'] == rnd]
                
                for chamber in ['Left', 'Right']:
                    c_df = rnd_df[rnd_df['chamber'] == chamber]
                    if c_df.empty: continue
                    lines.append(f"\n       [{chamber} Chamber]")
                    
                    for uc in config.USER_CATS:
                        u_df = c_df[c_df['user_cat'] == uc]
                        u_seen = u_df[u_df['action'] == 'seen']
                        u_share = u_df[u_df['action'] == 'shared']
                        
                        tot_seen = len(u_seen)
                        if tot_seen > 0:
                            dist = [len(u_seen[u_seen['item_cat'] == ic]) / tot_seen * 100 for ic in config.ITEM_CATS]
                            rates = [(len(u_share[u_share['item_cat'] == ic]) / len(u_seen[u_seen['item_cat'] == ic]) * 100) if len(u_seen[u_seen['item_cat'] == ic]) > 0 else 0.0 for ic in config.ITEM_CATS]
                            lines.append(f"         {uc:<12} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                            lines.append(f"         {uc:<12} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")
            lines.append("-" * 60)

    with open(os.path.join(sweep_dir, "Sweep_Summary.txt"), 'w') as f: f.write("\n".join(lines))
    print(f" -> Exported Sweep_Summary.txt")