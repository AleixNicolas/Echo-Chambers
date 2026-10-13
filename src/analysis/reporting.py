import pandas as pd
import numpy as np
from pathlib import Path
import config
from src.empirical import otree_parser
from src.analysis import event_logger, metrics

def calc_ordinal_slant_var(df_action):
    mapping = {'Left': -2, 'Center-Left': -1, 'Center': 0, 'Center-Right': 1, 'Right': 2}
    vals = df_action['item_cat'].map(mapping).dropna()
    if len(vals) == 0: return np.nan, np.nan
    return vals.mean(), vals.var(ddof=0)

def _format_share_matrix(df, index_col, index_cats, title):
    lines = ["\n" + "="*40, f" {title}", "="*40]
    grouped = df.groupby([index_col, 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
    for col in ['seen', 'shared']:
        if col not in grouped.columns: grouped[col] = 0
    grouped['share_rate'] = np.where(grouped['seen'] > 0, grouped['shared'] / grouped['seen'], 0.0)
    matrix = grouped.pivot(index=index_col, columns='item_cat', values='share_rate').fillna(0)
    matrix = matrix.reindex(index=index_cats, columns=config.ITEM_CATS).fillna(0)
    lines.append("\n" + matrix.to_string(float_format=lambda x: f"{x:.1%}"))
    return lines

def _build_core_report_lines(topic_df, max_round):
    lines = []
    
    user_action_counts = topic_df.groupby('user_id')['action'].value_counts().unstack(fill_value=0)
    if 'shared' not in user_action_counts.columns: user_action_counts['shared'] = 0
    zero_share_users = user_action_counts[user_action_counts['shared'] == 0].index.tolist()
    lines.append(f"\nZero-Share Participants : {len(zero_share_users)}")
    for uid in zero_share_users: lines.append(f"  - User ID: {uid}")
    
    multi_metrics = metrics.calculate_sourcing_metrics(topic_df)
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
            
            c_seen, c_share = c_df[c_df['action'] == 'seen'], c_df[c_df['action'] == 'shared']
            s_slant, s_var = calc_ordinal_slant_var(c_seen)
            lines.append(f"\n  [{chamber} Chamber] (Feed Slant: {s_slant:.2f} | Variance: {s_var:.2f})" if not np.isnan(s_slant) else f"\n  [{chamber} Chamber] (Feed Slant: N/A | Variance: N/A)")
            
            for uc in config.USER_CATS:
                u_seen, u_share = c_seen[c_seen['user_cat'] == uc], c_share[c_share['user_cat'] == uc]
                tot = len(u_seen)
                if tot > 0:
                    dist = [len(u_seen[u_seen['item_cat'] == ic]) / tot * 100 for ic in config.ITEM_CATS]
                    rates = [(len(u_share[u_share['item_cat'] == ic]) / len(u_seen[u_seen['item_cat'] == ic]) * 100) if len(u_seen[u_seen['item_cat'] == ic]) > 0 else 0 for ic in config.ITEM_CATS]
                    lines.append(f"    {uc:<12} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                    lines.append(f"    {uc:<12} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")

    lines.extend(_format_share_matrix(topic_df, 'user_cat', config.USER_CATS, "GLOBAL SHARE RATE MATRIX (5x5, ALL ROUNDS, BOTH CHAMBERS)"))
    
    topic_df_3x5 = topic_df.copy()
    topic_df_3x5['user_cat_3'] = topic_df_3x5['user_cat'].map({'Left': 'Left', 'Center-Left': 'Left', 'Center': 'Center', 'Center-Right': 'Right', 'Right': 'Right'})
    lines.extend(_format_share_matrix(topic_df_3x5, 'user_cat_3', ['Left', 'Center', 'Right'], "GLOBAL SHARE RATE MATRIX (3x5 COLLAPSED, ALL ROUNDS)"))

    for chamber in ['Left', 'Right']:
        chamber_df = topic_df[topic_df['chamber'] == chamber]
        if chamber_df.empty: continue
        chamber_df_3x5 = topic_df_3x5[topic_df_3x5['chamber'] == chamber]
        
        lines.extend(_format_share_matrix(chamber_df, 'user_cat', config.USER_CATS, f"{chamber.upper()} CHAMBER: SHARE RATE MATRIX (5x5)"))
        lines.extend(_format_share_matrix(chamber_df_3x5, 'user_cat_3', ['Left', 'Center', 'Right'], f"{chamber.upper()} CHAMBER: SHARE RATE MATRIX (3x5 COLLAPSED)"))
        
    return lines

def export_screening_combinations(phase1_csv_path, output_dir):
    phase1_csv_path = Path(phase1_csv_path)
    output_dir = Path(output_dir)
    
    if not phase1_csv_path.exists() or getattr(config, 'EMPIRICAL_TOPOLOGY_MODE', 'dual_cross_pressured') == 'single_topic': return

    df = pd.read_csv(phase1_csv_path, low_memory=False)
    
    def get_screening_opinion(row, topic):
        for c in [f'phase_1.1.player.{topic}_opinion_4', f'phase_1.1.player.{topic}_opinion_2', f'participant.{topic}_opinion_4', f'participant.{topic}_opinion_2']:
            if c in row and pd.notna(row[c]):
                try:
                    val = max(1.0, min(5.0, float(row[c])))
                    return (6.0 - val) if topic in config.INVERT_OPINIONS_FOR else val 
                except (ValueError, TypeError): pass
        return None 
        
    results = []
    for _, row in df.iterrows():
        c_val = get_screening_opinion(row, config.EMPIRICAL_SIDE_TOPIC)
        i_val = get_screening_opinion(row, config.EMPIRICAL_BASE_TOPIC)
        if c_val is not None and i_val is not None:
            results.append({'Secondary_Topic': otree_parser.get_emp_user_bucket(c_val), 'Base_Topic': otree_parser.get_emp_user_bucket(i_val)})
        
    res_df = pd.DataFrame(results)
    if res_df.empty: return
        
    cross_tab = pd.crosstab(res_df['Base_Topic'], res_df['Secondary_Topic']).reindex(index=config.USER_CATS, columns=config.USER_CATS, fill_value=0)
    
    lines = ["==========================================", " PHASE 1 SCREENING COMBINATIONS (5x5)", "==========================================\n"]
    lines.append(f"Total Phase 1 Participants Analyzed: {len(res_df)}\n")
    lines.append(f"Rows: Base Topic ({config.EMPIRICAL_BASE_TOPIC}) | Columns: Secondary Topic ({config.EMPIRICAL_SIDE_TOPIC})\n")
    lines.append(cross_tab.to_string())
    
    with open(output_dir / "Phase1_Screening_Combinations.txt", 'w') as f: 
        f.write("\n".join(lines))

def export_h3_report(event_log_df, output_dir, topics):
    if len(topics) < 2: return
    output_dir = Path(output_dir)
    base_t, side_t = topics[0], topics[1]
    
    users_df = event_log_df[['user_id', 'topic', 'user_cat', 'treatment']].drop_duplicates()
    pivot_cats = users_df.pivot(index='user_id', columns='topic', values='user_cat').reset_index()
    if base_t not in pivot_cats.columns or side_t not in pivot_cats.columns: return
    
    treat_map = users_df[['user_id', 'treatment']].drop_duplicates().set_index('user_id')['treatment'].to_dict()
    pivot_cats['treatment'] = pivot_cats['user_id'].map(treat_map)
    
    def get_2d_type(row):
        b_left = row[base_t] in ['Left', 'Center-Left']
        b_right = row[base_t] in ['Right', 'Center-Right']
        s_left = row[side_t] in ['Left', 'Center-Left']
        s_right = row[side_t] in ['Right', 'Center-Right']
        
        if b_left and s_left: return 'Type A (Pro-Imm, Climate-Left)'
        if b_left and s_right: return 'Type B (Pro-Imm, Climate-Right)'
        if b_right and s_left: return 'Type C (Anti-Imm, Climate-Left)'
        if b_right and s_right: return 'Type D (Anti-Imm, Climate-Right)'
        return 'Other'
        
    pivot_cats['Type'] = pivot_cats.apply(get_2d_type, axis=1)
    type_map = pivot_cats.set_index('user_id')['Type'].to_dict()
    
    side_df = event_log_df[event_log_df['topic'] == side_t].copy()
    side_df['Type'] = side_df['user_id'].map(type_map)
    
    lines = ["==========================================", f" H3 HYPOTHESIS REPORT: SIDE TOPIC ({side_t.upper()})", "==========================================\n"]
    lines.append("Comparing exact demographic 2D types across structural treatments to remove confounding.\n")
    
    for type_label in ['Type A (Pro-Imm, Climate-Left)', 'Type B (Pro-Imm, Climate-Right)', 'Type C (Anti-Imm, Climate-Left)', 'Type D (Anti-Imm, Climate-Right)']:
        type_df = side_df[side_df['Type'] == type_label]
        lines.append(f"--- {type_label.upper()} ---")
        
        if type_df.empty:
            lines.append("  No data available for this type.\n")
            continue
            
        def get_rate(df_sub):
            seen = len(df_sub[df_sub['action'] == 'seen'])
            shared = len(df_sub[df_sub['action'] == 'shared'])
            return (shared / seen * 100) if seen > 0 else np.nan, seen, shared
            
        seg_rate, seg_seen, seg_shared = get_rate(type_df[type_df['treatment'] == 'segregated'])
        int_rate, int_seen, int_shared = get_rate(type_df[type_df['treatment'] == 'integrated'])
        
        lines.append(f"  Segregated Share Rate : {seg_rate:>5.1f}%  (Shared {seg_shared} of {seg_seen} seen)")
        lines.append(f"  Integrated Share Rate : {int_rate:>5.1f}%  (Shared {int_shared} of {int_seen} seen)")
        
        if not np.isnan(seg_rate) and not np.isnan(int_rate):
            lines.append(f"  Effect (Seg - Int)    : {(seg_rate - int_rate):>5.1f}%\n")
        else:
            lines.append(f"  Effect (Seg - Int)    : N/A\n")
            
    with open(output_dir / "Report_H3_Analysis.txt", 'w') as f:
        f.write("\n".join(lines))

def export_comprehensive_summary(event_log_df, output_dir, is_empirical=False, phase2_csv_path=None, topics=['climate'], global_exclusions=None, round_exclusions=None, prefix=""):
    if event_log_df.empty: return
    output_dir = Path(output_dir)
    is_dual_topic = len(topics) > 1
    
    if is_empirical:
        df2 = pd.read_csv(phase2_csv_path, low_memory=False)
        user_aliases = {}
        for _, row in df2.iterrows():
            aliases = [str(row[c]).strip().lower() for c in ['participant.label', 'participant.code'] if c in df2.columns and pd.notna(row[c])]
            for alias in aliases: user_aliases[alias] = aliases
                
        clean_globals = [str(x).strip().lower() for x in global_exclusions] if global_exclusions else []
        valid_rows = []
        
        for r in range(1, min(config.ROUNDS, 5) + 1):
            part_col = f'phase_2.{r}.player.participated_this_round'
            active_users = df2[df2[part_col].astype(str).str.lower().isin(['1', '1.0', 'true', 'yes'])]['participant.code'].dropna().unique() if part_col in df2.columns else event_log_df[event_log_df['round'] == r]['user_id'].unique()
                
            clean_round_excl = [str(x).strip().lower() for x in round_exclusions.get(r, [])] if round_exclusions else []
            for uid in active_users:
                my_aliases = user_aliases.get(str(uid).strip().lower(), [str(uid).strip().lower()])
                if not any(a in clean_globals for a in my_aliases) and not any(a in clean_round_excl for a in my_aliases):
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
                cross_tab = pd.crosstab(pivot_cats[base_t], pivot_cats[side_t]).reindex(index=config.USER_CATS, columns=config.USER_CATS, fill_value=0)
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
            lines = ["==========================================", f" SUMMARY: {topic.upper()} | {treatment.upper()}", "==========================================\n"]
            
            if is_empirical:
                user_round_counts = valid_user_rounds[valid_user_rounds['user_id'].isin(all_topic_users)].groupby('user_id').size()
                completers = user_round_counts[user_round_counts >= (config.ROUNDS - 1)].index.tolist()
                dropouts = list(set(all_topic_users) - set(completers))
                
                lines.extend(["--- IN-NETWORK ATTRITION BY PROFILE ---", f"Total Participants : {len(all_topic_users)}", f"\nCompleted (>= 4 Rounds): {len(completers)}"])
                
                comp_df = base_topic_df[base_topic_df['user_id'].isin(completers)].drop_duplicates(subset=['user_id']).copy()
                profile_keys = ['LL', 'LC', 'LR', 'CL', 'CC', 'CR', 'RL', 'RC', 'RR'] if is_dual_topic else ['L', 'C', 'R']
                
                if not comp_df.empty:
                    comp_df['Profile'] = comp_df.apply(get_profile, axis=1)
                    c_counts = comp_df['Profile'].value_counts()
                else: c_counts = {}
                
                for quad in profile_keys: lines.append(f"  {quad.ljust(15)}: {c_counts.get(quad, 0)}")
                    
                lines.append(f"\nDropped Out: {len(dropouts)}")
                if len(dropouts) > 0:
                    dropout_df = base_topic_df[base_topic_df['user_id'].isin(dropouts)].drop_duplicates(subset=['user_id']).copy()
                    if not dropout_df.empty:
                        dropout_df['Profile'] = dropout_df.apply(get_profile, axis=1)
                        d_counts = dropout_df['Profile'].value_counts()
                    else: d_counts = {}
                    for quad in profile_keys: lines.append(f"  {quad.ljust(15)}: {d_counts.get(quad, 0)}")

            lines.extend(_build_core_report_lines(treat_df, min(config.ROUNDS, 5) if is_empirical else treat_df['round'].max()))
            
            out_filename = f"Empirical_Comprehensive_Summary_{treatment}.txt" if is_empirical else f"Simulation_Comprehensive_Summary_{prefix}_{treatment}.txt"
            with open(output_dir / out_filename, 'w') as f: 
                f.write("\n".join(lines))

def export_sweep_summary(sweep_aggregator, sweep_dir, k_val, regime_name, priority_str):
    sweep_dir = Path(sweep_dir)
    lines = ["==========================================", f" SIMULATION SWEEP SUMMARY", f" Regime: {regime_name} | Priority: {priority_str} | K: {k_val}", "==========================================\n"]
    
    for struct in ['integrated', 'segregated']:
        lines.append(f"=== {struct.upper()} TOPOLOGY ===")
        for n_val in sorted(sweep_aggregator.keys()):
            lines.append(f"\n[ N = {n_val} ]")
            t_df = event_logger.build_simulation_event_log({struct: sweep_aggregator[n_val][struct]})
            
            for topic in getattr(config, 'SIM_TOPICS', ['base_topic', 'side_topic']):
                topic_df = t_df[t_df['topic'] == topic] if 'topic' in t_df.columns else t_df
                if topic_df.empty: continue
                
                lines.append(f"\n  --- TOPIC: {topic.upper()} ---")
                for rnd in range(1, config.ROUNDS + 1):
                    lines.append(f"\n    -- ROUND {rnd} --")
                    for chamber in ['Left', 'Right']:
                        c_df = topic_df[(topic_df['round'] == rnd) & (topic_df['chamber'] == chamber)]
                        if c_df.empty: continue
                        s_slant, s_var = calc_ordinal_slant_var(c_df[c_df['action'] == 'seen'])
                        lines.append(f"         [{chamber} Chamber] (Feed Slant: {s_slant:.2f} | Variance: {s_var:.2f})" if not np.isnan(s_slant) else f"         [{chamber} Chamber] (Feed Slant: N/A | Variance: N/A)")

    with open(sweep_dir / "Sweep_Summary.txt", 'w') as f: 
        f.write("\n".join(lines))
        
def export_h3_pooled_regression(event_log_df, output_dir, topics):
    if len(topics) < 2: return
    output_dir = Path(output_dir)
    
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        with open(output_dir / "Report_H3_Pooled_Regression.txt", 'w') as f:
            f.write("ERROR: 'statsmodels' library is not installed.\nRun 'pip install statsmodels' to generate the H3 logistic regression report.")
        return

    base_t, side_t = topics[0], topics[1]
    
    # 1. Build 2D User Profiles
    users_df = event_log_df[['user_id', 'topic', 'user_cat', 'treatment']].drop_duplicates()
    pivot_cats = users_df.pivot(index=['user_id', 'treatment'], columns='topic', values='user_cat').reset_index()
    if base_t not in pivot_cats.columns or side_t not in pivot_cats.columns: return
    
    def get_2d_type(row):
        b_left = row[base_t] in ['Left', 'Center-Left']
        b_right = row[base_t] in ['Right', 'Center-Right']
        s_left = row[side_t] in ['Left', 'Center-Left']
        s_right = row[side_t] in ['Right', 'Center-Right']
        
        if b_left and s_left: return 'Type_A_ProImm_ClimLeft'
        if b_left and s_right: return 'Type_B_ProImm_ClimRight'
        if b_right and s_left: return 'Type_C_AntiImm_ClimLeft'
        if b_right and s_right: return 'Type_D_AntiImm_ClimRight'
        return 'Other'
        
    pivot_cats['Type'] = pivot_cats.apply(get_2d_type, axis=1)
    type_map = pivot_cats.set_index('user_id')['Type'].to_dict()
    
    # 2. Build the Item-Level Exposure Dataset
    side_events = event_log_df[event_log_df['topic'] == side_t]
    seen_df = side_events[side_events['action'] == 'seen'].copy()
    shared_df = side_events[side_events['action'] == 'shared']
    
    # Map shared events (1 if shared, 0 if not)
    shared_keys = set(zip(shared_df['user_id'], shared_df['round'], shared_df['item_id']))
    seen_df['is_shared'] = seen_df.apply(lambda r: 1 if (r['user_id'], r['round'], r['item_id']) in shared_keys else 0, axis=1)
    
    seen_df['Type'] = seen_df['user_id'].map(type_map)
    seen_df = seen_df[seen_df['Type'] != 'Other'].copy()
    
    lines = ["==========================================", f" H3 POOLED LOGISTIC REGRESSION: {side_t.upper()}", "==========================================\n"]
    lines.append("Unit of Analysis: Item-Level Exposure (Seen Events)")
    lines.append("Dependent Variable: is_shared (0 or 1)")
    lines.append("Standard Errors: Clustered at the User_ID level\n")
    lines.append(f"Total Exposure Events Analyzed: {len(seen_df)}\n")
    
    # 3. Fit the Mixed-Effects Logistic Model
    try:
        # We interact Treatment with 2D Type, and control for the Leaning of the Item itself
        formula = "is_shared ~ C(treatment, Treatment(reference='integrated')) * C(Type) + C(item_cat)"
        model = smf.logit(formula, data=seen_df)
        # Cluster standard errors by user to account for repeat-sharing behavior
        result = model.fit(cov_type='cluster', cov_kwds={'groups': seen_df['user_id']}, disp=0)
        
        lines.append(result.summary().as_text())
        
        lines.append("\n\n=== HOW TO READ THIS FOR H3 ===")
        lines.append("Look at the interaction terms (e.g., 'C(treatment...)[T.segregated]:C(Type)[T.Type_C...]').")
        lines.append("A positive, significant coefficient (P>|z| < 0.05) here means that being placed in the")
        lines.append("Segregated network significantly INCREASES the probability of that specific cross-pressured")
        lines.append("demographic sharing an item, compared to the exact same demographic in the Integrated network.")
        
    except Exception as e:
        lines.append(f"Regression failed to converge (likely due to sparse pilot data). Error: {str(e)}")

    with open(output_dir / "Report_H3_Pooled_Regression.txt", 'w') as f:
        f.write("\n".join(lines))