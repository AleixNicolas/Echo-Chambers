import pandas as pd
import numpy as np
from pathlib import Path
import config
from src import data_wrangle, stats_suite

def calc_ordinal_slant_var(df_action):
    mapping = {'Left': -2, 'Center-Left': -1, 'Center': 0, 'Center-Right': 1, 'Right': 2}
    vals = df_action['item_cat'].map(mapping).dropna()
    if len(vals) == 0: return np.nan, np.nan
    return vals.mean(), vals.var(ddof=0)

def export_screening_combinations(phase1_csv_path, output_dir):
    phase1_csv_path = Path(phase1_csv_path)
    output_dir = Path(output_dir)
    
    if not phase1_csv_path.exists(): return
    if getattr(config, 'EMPIRICAL_TOPOLOGY_MODE', 'dual_cross_pressured') == 'single_topic' or not getattr(config, 'EMPIRICAL_SIDE_TOPIC', None): return

    df = pd.read_csv(phase1_csv_path, low_memory=False)
    
    def get_screening_opinion(row, topic):
        cols = [f'phase_1.1.player.{topic}_opinion_4', f'phase_1.1.player.{topic}_opinion_2', 
                f'phase_1.1.player.{topic}_opinion_1', f'{topic}_opinion_4', f'{topic}_opinion_2', 
                f'participant.{topic}_opinion_4', f'participant.{topic}_opinion_2']
        for c in cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip() != '':
                try:
                    val = max(1.0, min(5.0, float(row[c])))
                    return (6.0 - val) if topic in config.INVERT_OPINIONS_FOR else val
                except: pass
        return None 
        
    results = []
    for _, row in df.iterrows():
        c_val = get_screening_opinion(row, config.EMPIRICAL_SIDE_TOPIC)
        i_val = get_screening_opinion(row, config.EMPIRICAL_BASE_TOPIC)
        if c_val is not None and i_val is not None:
            results.append({
                'Secondary_Topic': data_wrangle.get_emp_user_bucket(c_val), 
                'Base_Topic': data_wrangle.get_emp_user_bucket(i_val)
            })
        
    res_df = pd.DataFrame(results)
    if res_df.empty: return
        
    cross_tab = pd.crosstab(res_df['Base_Topic'], res_df['Secondary_Topic'])
    cross_tab = cross_tab.reindex(index=config.USER_CATS, columns=config.USER_CATS, fill_value=0)
    
    lines = ["==========================================", " PHASE 1 SCREENING COMBINATIONS (5x5)", "==========================================\n"]
    lines.append(f"Total Phase 1 Participants Analyzed: {len(res_df)}\n")
    lines.append(f"Rows: Base Topic ({config.EMPIRICAL_BASE_TOPIC}) | Columns: Secondary Topic ({config.EMPIRICAL_SIDE_TOPIC})\n")
    lines.append(cross_tab.to_string())
    
    with open(output_dir / "Phase1_Screening_Combinations.txt", 'w') as f: 
        f.write("\n".join(lines))

def _build_core_report_lines(topic_df, max_round):
    lines = []
    
    # 1. Zero-Share tracking
    user_action_counts = topic_df.groupby('user_id')['action'].value_counts().unstack(fill_value=0)
    if 'shared' not in user_action_counts.columns: user_action_counts['shared'] = 0
    zero_share_users = user_action_counts[user_action_counts['shared'] == 0].index.tolist()
    lines.append(f"\nZero-Share Participants : {len(zero_share_users)}")
    if zero_share_users:
        for uid in zero_share_users: lines.append(f"  - User ID: {uid}")
    lines.append("")
    
    # 2. Round-by-round breakdown
    multi_metrics = stats_suite.calculate_sourcing_metrics(topic_df)
    src_metrics = multi_metrics.get('Global', {}) if multi_metrics else {}
    def fmt(val): return f"{val:.2f}" if not np.isnan(val) else "N/A "

    for r in range(1, max_round + 1):
        lines.append(f"\n--- ROUND {r} ---")
        r_df = topic_df[topic_df['round'] == r]
        lines.append(f"  Active Participants: {r_df['user_id'].nunique()}")
        
        if src_metrics and r <= len(src_metrics['rounds']):
            r_idx = r - 1
            lines.append(f"  Sourcing Averages : {fmt(src_metrics['rep_avg'][r_idx])} Repeats | {fmt(src_metrics['pool_avg'][r_idx])} Pool | {fmt(src_metrics['back_avg'][r_idx])} Backlog")

        for chamber in ['Left', 'Right']:
            c_df = r_df[r_df['chamber'] == chamber]
            if c_df.empty: continue
            
            c_seen = c_df[c_df['action'] == 'seen']
            c_share = c_df[c_df['action'] == 'shared']
            
            s_slant, s_var = calc_ordinal_slant_var(c_seen)
            slant_str = f"{s_slant:.2f}" if not np.isnan(s_slant) else "N/A"
            var_str = f"{s_var:.2f}" if not np.isnan(s_var) else "N/A"
            lines.append(f"\n  [{chamber} Chamber] (Feed Slant: {slant_str} | Variance: {var_str})")
            
            for uc in config.USER_CATS:
                u_seen = c_seen[c_seen['user_cat'] == uc]
                u_share = c_share[c_share['user_cat'] == uc]
                tot_cat_seen = len(u_seen)
                if tot_cat_seen > 0:
                    dist = [len(u_seen[u_seen['item_cat'] == ic]) / tot_cat_seen * 100 for ic in config.ITEM_CATS]
                    rates = [(len(u_share[u_share['item_cat'] == ic]) / len(u_seen[u_seen['item_cat'] == ic]) * 100) if len(u_seen[u_seen['item_cat'] == ic]) > 0 else 0 for ic in config.ITEM_CATS]
                    lines.append(f"    {uc:<12} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                    lines.append(f"    {uc:<12} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")

    # 3. Overall share rate
    lines.append("\n" + "="*40 + "\n OVERALL SHARE RATE BY USER CATEGORY\n" + "="*40)
    user_cat_grouped = topic_df.groupby(['user_cat', 'action']).size().unstack(fill_value=0)
    for col in ['seen', 'shared']:
        if col not in user_cat_grouped.columns: user_cat_grouped[col] = 0
    user_cat_grouped['share_rate'] = np.where(user_cat_grouped['seen'] > 0, user_cat_grouped['shared'] / user_cat_grouped['seen'], 0.0)
    user_cat_grouped = user_cat_grouped.reindex(config.USER_CATS).fillna(0)
    
    for uc in config.USER_CATS:
        seen_val = int(user_cat_grouped.loc[uc, 'seen'])
        shared_val = int(user_cat_grouped.loc[uc, 'shared'])
        rate_val = user_cat_grouped.loc[uc, 'share_rate']
        lines.append(f"  {uc:<14}: {rate_val:>6.1%}  ({shared_val} shared / {seen_val} seen)")

    # 4. Global matrix 5x5
    lines.append("\n" + "="*40 + "\n GLOBAL SHARE RATE MATRIX (5x5, ALL ROUNDS, BOTH CHAMBERS)\n" + "="*40)
    glob_grouped = topic_df.groupby(['user_cat', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
    for col in ['seen', 'shared']:
        if col not in glob_grouped.columns: glob_grouped[col] = 0
    glob_grouped['share_rate'] = np.where(glob_grouped['seen'] > 0, glob_grouped['shared'] / glob_grouped['seen'], 0.0)
    glob_matrix = glob_grouped.pivot(index='user_cat', columns='item_cat', values='share_rate').fillna(0)
    glob_matrix = glob_matrix.reindex(index=config.USER_CATS, columns=config.ITEM_CATS).fillna(0)
    lines.append("\n" + glob_matrix.to_string(float_format=lambda x: f"{x:.1%}"))

    # 5. Global matrix 3x5 Collapsed (3 User Categories x 5 Item Categories)
    lines.append("\n\n" + "="*40 + "\n GLOBAL SHARE RATE MATRIX (3x5 COLLAPSED, ALL ROUNDS)\n" + "="*40)
    
    cat_map = {'Left': 'Left', 'Center-Left': 'Left', 'Center': 'Center', 'Center-Right': 'Right', 'Right': 'Right'}
    topic_df_3x5 = topic_df.copy()
    topic_df_3x5['user_cat_3'] = topic_df_3x5['user_cat'].map(cat_map)
    
    glob_grouped_3x5 = topic_df_3x5.groupby(['user_cat_3', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
    for col in ['seen', 'shared']:
        if col not in glob_grouped_3x5.columns: glob_grouped_3x5[col] = 0
    glob_grouped_3x5['share_rate'] = np.where(glob_grouped_3x5['seen'] > 0, glob_grouped_3x5['shared'] / glob_grouped_3x5['seen'], 0.0)
    
    glob_matrix_3x5 = glob_grouped_3x5.pivot(index='user_cat_3', columns='item_cat', values='share_rate').fillna(0)
    glob_matrix_3x5 = glob_matrix_3x5.reindex(index=['Left', 'Center', 'Right'], columns=config.ITEM_CATS).fillna(0)
    lines.append("\n" + glob_matrix_3x5.to_string(float_format=lambda x: f"{x:.1%}"))

    # 6. Chamber matrices
    for chamber in ['Left', 'Right']:
        chamber_df = topic_df[topic_df['chamber'] == chamber]
        if chamber_df.empty: continue
        
        # 5x5 Chamber output
        lines.append("\n\n" + "="*40 + f"\n {chamber.upper()} CHAMBER: SHARE RATE MATRIX (5x5)\n" + "="*40)
        grouped = chamber_df.groupby(['user_cat', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
        for col in ['seen', 'shared']:
            if col not in grouped.columns: grouped[col] = 0
        grouped['share_rate'] = np.where(grouped['seen'] > 0, grouped['shared'] / grouped['seen'], 0.0)
        matrix = grouped.pivot(index='user_cat', columns='item_cat', values='share_rate').fillna(0)
        matrix = matrix.reindex(index=config.USER_CATS, columns=config.ITEM_CATS).fillna(0)
        lines.append("\n" + matrix.to_string(float_format=lambda x: f"{x:.1%}"))
        
        # 3x5 Chamber output
        lines.append("\n\n" + "="*40 + f"\n {chamber.upper()} CHAMBER: SHARE RATE MATRIX (3x5 COLLAPSED)\n" + "="*40)
        chamber_df_3x5 = topic_df_3x5[topic_df_3x5['chamber'] == chamber]
        grouped_3x5 = chamber_df_3x5.groupby(['user_cat_3', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
        for col in ['seen', 'shared']:
            if col not in grouped_3x5.columns: grouped_3x5[col] = 0
        grouped_3x5['share_rate'] = np.where(grouped_3x5['seen'] > 0, grouped_3x5['shared'] / grouped_3x5['seen'], 0.0)
        matrix_3x5 = grouped_3x5.pivot(index='user_cat_3', columns='item_cat', values='share_rate').fillna(0)
        matrix_3x5 = matrix_3x5.reindex(index=['Left', 'Center', 'Right'], columns=config.ITEM_CATS).fillna(0)
        lines.append("\n" + matrix_3x5.to_string(float_format=lambda x: f"{x:.1%}"))
        
    return lines

def export_comprehensive_summary(event_log_df, output_dir, is_empirical=False, phase2_csv_path=None, topics=['climate'], global_exclusions=None, round_exclusions=None, prefix=""):
    if event_log_df.empty: return
    output_dir = Path(output_dir)
    is_dual_topic = len(topics) > 1
    
    if is_empirical:
        df2 = pd.read_csv(phase2_csv_path, low_memory=False)
        user_aliases = {}
        for _, row in df2.iterrows():
            aliases = []
            if 'participant.label' in df2.columns and pd.notna(row['participant.label']): 
                aliases.append(str(row['participant.label']).strip().lower())
            if 'participant.code' in df2.columns and pd.notna(row['participant.code']): 
                aliases.append(str(row['participant.code']).strip().lower())
            for alias in aliases: user_aliases[alias] = aliases
                
        clean_globals = [str(x).strip().lower() for x in global_exclusions] if global_exclusions else []
        valid_rows = []
        max_interactive_round = min(config.ROUNDS, 5)
        
        for r in range(1, max_interactive_round + 1):
            part_col = f'phase_2.{r}.player.participated_this_round'
            if part_col in df2.columns and 'participant.code' in df2.columns:
                is_active = df2[part_col].apply(lambda x: str(x).strip().lower() in ['1', '1.0', 'true', 'yes'])
                active_users = df2[is_active]['participant.code'].dropna().unique()
            else:
                active_users = event_log_df[event_log_df['round'] == r]['user_id'].unique()
                
            clean_round_excl = [str(x).strip().lower() for x in round_exclusions.get(r, [])] if round_exclusions else []
            for uid in active_users:
                clean_uid = str(uid).strip().lower()
                my_aliases = user_aliases.get(clean_uid, [clean_uid])
                if not any(alias in clean_globals for alias in my_aliases) and not any(alias in clean_round_excl for alias in my_aliases):
                    valid_rows.append({'user_id': str(uid).upper(), 'round': r})
                    
        valid_user_rounds = pd.DataFrame(valid_rows).drop_duplicates()
        active_event_log_df = pd.merge(event_log_df, valid_user_rounds, on=['user_id', 'round'], how='inner')
        active_event_log_df = active_event_log_df[active_event_log_df['round'] <= 5]

        user_2d_profiles = {}
        if is_dual_topic:
            base_t, side_t = topics[0], topics[1]
            user_topic_cats = active_event_log_df[['user_id', 'topic', 'user_cat']].drop_duplicates()
            pivot_cats = user_topic_cats.pivot(index='user_id', columns='topic', values='user_cat')
            
            for uid, row in pivot_cats.iterrows():
                b_cat = str(row.get(base_t, 'Center'))
                b_char = 'L' if 'Left' in b_cat else ('R' if 'Right' in b_cat else 'C')
                s_cat = str(row.get(side_t, 'Center'))
                s_char = 'L' if 'Left' in s_cat else ('R' if 'Right' in s_cat else 'C')
                user_2d_profiles[uid] = f"{b_char}{s_char}"
                
            if base_t in pivot_cats.columns and side_t in pivot_cats.columns:
                cross_tab = pd.crosstab(pivot_cats[base_t], pivot_cats[side_t])
                cross_tab = cross_tab.reindex(index=config.USER_CATS, columns=config.USER_CATS, fill_value=0)
                cohort_lines = ["==========================================", " ACTUAL PHASE 2 COHORT COMBINATIONS (5x5)", "==========================================\n"]
                cohort_lines.append(f"Total Active Phase 2 In-Network Participants: {len(pivot_cats)}\n")
                cohort_lines.append(f"Rows: Base Topic ({base_t}) | Columns: Secondary Topic ({side_t})\n")
                cohort_lines.append(cross_tab.to_string())
                with open(output_dir / "Phase2_Cohort_Combinations.txt", 'w') as f: 
                    f.write("\n".join(cohort_lines))

        def get_profile(row):
            if is_dual_topic: return user_2d_profiles.get(row['user_id'], "Unknown")
            ucat = row.get('user_cat', 'Center')
            return 'L' if 'Left' in ucat else ('R' if 'Right' in ucat else 'C')

    for topic in topics:
        if is_empirical:
            base_topic_df = event_log_df[(event_log_df['topic'] == topic) & (event_log_df['round'] <= 5)] if 'topic' in event_log_df.columns else event_log_df[event_log_df['round'] <= 5]
            all_topic_users = base_topic_df['user_id'].unique()
            topic_df = active_event_log_df[active_event_log_df['topic'] == topic] if 'topic' in active_event_log_df.columns else active_event_log_df
        else:
            topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df

        if topic_df.empty: continue

        for treatment in topic_df['treatment'].unique():
            treat_df = topic_df[topic_df['treatment'] == treatment]
            header = f" EMPIRICAL SUMMARY: {topic.upper()} | {treatment.upper()}" if is_empirical else f" SIMULATION SUMMARY: {topic.upper()} | {treatment.upper()}"
            lines = ["==========================================", header, "==========================================\n"]
            
            if is_empirical:
                user_round_counts = valid_user_rounds[valid_user_rounds['user_id'].isin(all_topic_users)].groupby('user_id').size()
                completers = user_round_counts[user_round_counts >= (config.ROUNDS - 1)].index.tolist()
                dropouts = list(set(all_topic_users) - set(completers))
                
                lines.append("--- IN-NETWORK ATTRITION BY PROFILE ---")
                lines.append(f"Total In-Network Participants : {len(all_topic_users)}")
                lines.append(f"\nCompleted Valid Session (>= 4 Rounds): {len(completers)}")
                
                comp_df = base_topic_df[base_topic_df['user_id'].isin(completers)].drop_duplicates(subset=['user_id']).copy()
                profile_keys = ['LL', 'LC', 'LR', 'CL', 'CC', 'CR', 'RL', 'RC', 'RR'] if is_dual_topic else ['L', 'C', 'R']
                
                if not comp_df.empty:
                    comp_df['Profile'] = comp_df.apply(get_profile, axis=1)
                    c_counts = comp_df['Profile'].value_counts()
                else: c_counts = {}
                    
                for quad in profile_keys: lines.append(f"  {quad.ljust(15)}: {c_counts.get(quad, 0)}")
                    
                lines.append(f"\nDropped Out (Missed >= 2 Rounds): {len(dropouts)}")
                if len(dropouts) > 0:
                    dropout_df = base_topic_df[base_topic_df['user_id'].isin(dropouts)].drop_duplicates(subset=['user_id']).copy()
                    if not dropout_df.empty:
                        dropout_df['Profile'] = dropout_df.apply(get_profile, axis=1)
                        d_counts = dropout_df['Profile'].value_counts()
                    else: d_counts = {}
                    for quad in profile_keys: lines.append(f"  {quad.ljust(15)}: {d_counts.get(quad, 0)}")

            max_round = min(config.ROUNDS, 5) if is_empirical else treat_df['round'].max()
            lines.extend(_build_core_report_lines(treat_df, max_round))
            
            if is_empirical:
                safe_suffix = f"_{topic}_{treatment}" if is_dual_topic else f"_{treatment}"
                out_filename = f"Empirical_Comprehensive_Summary{safe_suffix}.txt"
            else:
                safe_prefix = f"{prefix}_{treatment}_{topic}" if is_dual_topic else f"{prefix}_{treatment}"
                out_filename = f"Simulation_Comprehensive_Summary_{safe_prefix}.txt"
                
            with open(output_dir / out_filename, 'w') as f: 
                f.write("\n".join(lines))

def export_sweep_summary(sweep_aggregator, sweep_dir, k_val, regime_name, priority_str):
    sweep_dir = Path(sweep_dir)
    lines = ["==========================================", f" SIMULATION SWEEP SUMMARY", f" Regime: {regime_name} | Priority: {priority_str} | K: {k_val}", "==========================================\n"]
    sim_topics = getattr(config, 'SIM_TOPICS', ['base_topic', 'side_topic'])
    
    for struct in ['integrated', 'segregated']:
        lines.append(f"=== {struct.upper()} TOPOLOGY ===")
        for n_val in sorted(sweep_aggregator.keys()):
            lines.append(f"\n[ N = {n_val} ]")
            data = sweep_aggregator[n_val][struct]
            t_df = data_wrangle.build_simulation_event_log({struct: data})
            
            for topic in sim_topics:
                topic_df = t_df[t_df['topic'] == topic] if 'topic' in t_df.columns else t_df
                if topic_df.empty: continue
                
                lines.append(f"\n  --- TOPIC: {topic.upper()} ---")
                multi_metrics = stats_suite.calculate_sourcing_metrics(topic_df)
                src_metrics = multi_metrics.get('Global', {}) if multi_metrics else {}
                def fmt(val): return f"{val:.2f}" if not np.isnan(val) else "N/A "
                
                for rnd in range(1, config.ROUNDS + 1):
                    lines.append(f"\n    -- ROUND {rnd} --")
                    if src_metrics and rnd <= len(src_metrics['rounds']):
                        r_idx = rnd - 1
                        lines.append(f"         Sourcing Averages : {fmt(src_metrics['rep_avg'][r_idx])} Repeats | {fmt(src_metrics['pool_avg'][r_idx])} Pool | {fmt(src_metrics['back_avg'][r_idx])} Backlog")
                    
                    rnd_df = topic_df[topic_df['round'] == rnd]
                    
                    for chamber in ['Left', 'Right']:
                        c_df = rnd_df[rnd_df['chamber'] == chamber]
                        if c_df.empty: continue
                        
                        c_seen = c_df[c_df['action'] == 'seen']
                        s_slant, s_var = calc_ordinal_slant_var(c_seen)
                        
                        # The actual fix: splitting the format logic out of the f-string
                        slant_str = f"{s_slant:.2f}" if not np.isnan(s_slant) else "N/A"
                        var_str = f"{s_var:.2f}" if not np.isnan(s_var) else "N/A"
                        
                        lines.append(f"\n         [{chamber} Chamber] (Feed Slant: {slant_str} | Variance: {var_str})")
                        
                        for uc in config.USER_CATS:
                            u_df = c_df[c_df['user_cat'] == uc]
                            if u_df.empty: continue 
                            
                            u_seen = u_df[u_df['action'] == 'seen']
                            u_share = u_df[u_df['action'] == 'shared']
                            tot_seen = len(u_seen)
                            if tot_seen > 0:
                                dist = [len(u_seen[u_seen['item_cat'] == ic]) / tot_seen * 100 for ic in config.ITEM_CATS]
                                rates = [(len(u_share[u_share['item_cat'] == ic]) / len(u_seen[u_seen['item_cat'] == ic]) * 100) if len(u_seen[u_seen['item_cat'] == ic]) > 0 else 0.0 for ic in config.ITEM_CATS]
                                lines.append(f"           {uc:<14} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                                lines.append(f"           {uc:<14} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")
            lines.append("-" * 60)

    with open(sweep_dir / "Sweep_Summary.txt", 'w') as f: 
        f.write("\n".join(lines))

    with open(sweep_dir / "Sweep_Summary.txt", 'w') as f: 
        f.write("\n".join(lines))