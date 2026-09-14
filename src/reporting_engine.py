import os
import pandas as pd
import numpy as np
import config
from src import data_wrangle, stats_suite

def export_screening_combinations(phase1_csv_path, output_dir):
    if not os.path.exists(phase1_csv_path):
        print(f"[!] Warning: Phase 1 CSV {phase1_csv_path} not found. Skipping screening combinations report.")
        return
    
    df = pd.read_csv(phase1_csv_path)
    
    def get_screening_opinion(row, topic):
        # Look for Phase 1 opinion indicators using the correct oTree column prefixes
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
                    if topic in ['imm', 'immigration']:
                        return 6.0 - val
                    return val
                except:
                    pass
        return None # Return None instead of defaulting to 3.0 to skip empty rows
    
    def bucket_val(val):
        if val > 3.0: return 'Left'
        elif val < 3.0: return 'Right'
        else: return 'Neutral'
        
    results = []
    for _, row in df.iterrows():
        c_val = get_screening_opinion(row, 'climate')
        i_val = get_screening_opinion(row, 'imm')
        
        # Only process rows that have valid data for both topics (filters out dropouts)
        if c_val is not None and i_val is not None:
            results.append({
                'Climate': bucket_val(c_val),
                'Immigration': bucket_val(i_val)
            })
        
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("[!] Warning: No valid screening data found to generate combinations.")
        return
        
    cross_tab = pd.crosstab(res_df['Climate'], res_df['Immigration'])
    
    # Ensure all categories appear in the matrix even if the count is 0
    order = ['Left', 'Neutral', 'Right']
    cross_tab = cross_tab.reindex(index=order, columns=order, fill_value=0)
    
    lines = ["==========================================", " PHASE 1 SCREENING COMBINATIONS", "==========================================\n"]
    lines.append("Counts of participants by their leaning across both topics:")
    lines.append(f"Total Phase 1 Participants Analyzed: {len(res_df)}\n")
    lines.append(cross_tab.to_string())
    lines.append("\nNote: 'Neutral' strictly represents an opinion score of 3.0.")
    
    file_path = os.path.join(output_dir, "Phase1_Screening_Combinations.txt")
    with open(file_path, 'w') as f:
        f.write("\n".join(lines))
    print(f" -> Exported Phase1_Screening_Combinations.txt")

def export_comprehensive_empirical_summary(event_log_df, output_dir, phase2_csv_path, topics=['climate'], global_exclusions=None, round_exclusions=None):
    if event_log_df.empty:
        print("[!] Warning: Event log is empty. Skipping empirical summary report.")
        return

    is_dual = len(topics) > 1
    df2 = pd.read_csv(phase2_csv_path)
    
    # ---------------------------------------------------------
    # 1. ALIAS MAPPING & PHASE 1 MERGE FOR OPINIONS
    # ---------------------------------------------------------
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
    
    # ---------------------------------------------------------
    # 2. FILTER: True Participation based on oTree Phase 2 Data
    # ---------------------------------------------------------
    id_col = 'participant.label' if 'participant.label' in df2.columns else 'participant.code'
    valid_rows = []
    
    max_interactive_round = min(config.ROUNDS, 5)
    
    for r in range(1, max_interactive_round + 1):
        check_col = f'network.{r}.player.id_in_group'
        
        if check_col in df2.columns:
            active_users = df2[df2[check_col].notna()][id_col].dropna().unique()
        else:
            active_users = event_log_df[event_log_df['round'] == r]['user_id'].unique()
            
        cleaned_active_users = []
        clean_round_excl = [str(x).strip().lower() for x in round_exclusions.get(r, [])] if round_exclusions else []
        
        for uid in active_users:
            clean_uid = str(uid).strip().lower()
            my_aliases = user_aliases.get(clean_uid, [clean_uid])
            
            is_global_excluded = any(alias in clean_globals for alias in my_aliases)
            is_round_excluded = any(alias in clean_round_excl for alias in my_aliases)
            
            if is_global_excluded:
                continue
            elif is_round_excluded:
                continue
            else:
                cleaned_active_users.append(uid)
                
        for uid in cleaned_active_users:
            valid_rows.append({'user_id': uid, 'round': r})
            
    valid_user_rounds = pd.DataFrame(valid_rows).drop_duplicates()
    active_event_log_df = pd.merge(event_log_df, valid_user_rounds, on=['user_id', 'round'], how='inner')
    active_event_log_df = active_event_log_df[active_event_log_df['round'] <= 5]
    
    for topic in topics:
        base_topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        base_topic_df = base_topic_df[base_topic_df['round'] <= 5] 
        all_topic_users = base_topic_df['user_id'].unique()
        
        topic_df = active_event_log_df[active_event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else active_event_log_df
        
        lines = ["==========================================", f" EMPIRICAL COMPREHENSIVE SUMMARY: {topic.upper()} ", "==========================================\n"]
        
        # ---------------------------------------------------------
        # 3. ATTRITION & DEMOGRAPHICS (PHASE 2 FOCUSED)
        # ---------------------------------------------------------
        user_round_counts = valid_user_rounds[valid_user_rounds['user_id'].isin(all_topic_users)].groupby('user_id').size()
        
        completers = user_round_counts[user_round_counts >= 4].index.tolist()
        dropouts = list(set(all_topic_users) - set(completers))
        
        lines.append("--- ATTRITION & DEMOGRAPHICS ---")
        lines.append(f"Total Phase 2 Participants : {len(all_topic_users)}")
        
        lines.append(f"\nCompleted >= 4 Rounds      : {len(completers)}")
        lines.append("Leaning of Completers:")
        comp_df = base_topic_df[base_topic_df['user_id'].isin(completers)].drop_duplicates(subset=['user_id'])
        c_counts = comp_df['user_cat'].value_counts()
        for cat in config.USER_CATS:
            lines.append(f"  {cat.ljust(15)}: {c_counts.get(cat, 0)}")
            
        lines.append(f"\nDrop-outs (< 4 Rounds)     : {len(dropouts)}")
        if len(dropouts) > 0:
            lines.append("Leaning of Drop-outs:")
            dropout_df = base_topic_df[base_topic_df['user_id'].isin(dropouts)].drop_duplicates(subset=['user_id'])
            d_counts = dropout_df['user_cat'].value_counts()
            for cat in config.USER_CATS:
                lines.append(f"  {cat.ljust(15)}: {d_counts.get(cat, 0)}")
                
        # ---------------------------------------------------------
        # ZERO-SHARE PARTICIPANTS TRACKING
        # ---------------------------------------------------------
        user_action_counts = topic_df.groupby('user_id')['action'].value_counts().unstack(fill_value=0)
        if 'shared' not in user_action_counts.columns:
            user_action_counts['shared'] = 0
            
        zero_share_users = user_action_counts[user_action_counts['shared'] == 0].index.tolist()
        
        lines.append(f"\nZero-Share Participants    : {len(zero_share_users)}")
        if len(zero_share_users) > 0:
            lines.append("Details of Zero-Share Participants:")
            for uid in zero_share_users:
                u_row = topic_df[topic_df['user_id'] == uid].iloc[0]
                op_raw = u_row.get('op_raw', 'N/A')
                cat = u_row.get('user_cat', 'N/A')
                lines.append(f"  - User ID: {uid} | Leaning: {cat} | Baseline Opinion Score: {op_raw}")
        lines.append("")
        
        # ---------------------------------------------------------
        # 4. GLOBAL SHARE RATES
        # ---------------------------------------------------------
        lines.append("--- GLOBAL SHARE RATES ---")
        tot_seen = len(topic_df[topic_df['action'] == 'seen'])
        tot_shared = len(topic_df[topic_df['action'] == 'shared'])
        gen_rate = (tot_shared / tot_seen * 100) if tot_seen > 0 else 0.0
        
        lines.append(f"Overall Share Rate: {gen_rate:.1f}% ({tot_shared} total items shared / {tot_seen} total items seen)")
        
        lines.append("\nShare Rate by Participant Leaning (Any News):")
        for cat in config.USER_CATS:
            cat_seen = len(topic_df[(topic_df['user_cat'] == cat) & (topic_df['action'] == 'seen')])
            cat_shared = len(topic_df[(topic_df['user_cat'] == cat) & (topic_df['action'] == 'shared')])
            cat_rate = (cat_shared / cat_seen * 100) if cat_seen > 0 else 0.0
            lines.append(f"  {cat.ljust(15)}: {cat_rate:>5.1f}% ({cat_shared} shared / {cat_seen} seen)")
        lines.append("")

        # ---------------------------------------------------------
        # 4B. SHARE RATES BY CONTRARY NEIGHBORS
        # ---------------------------------------------------------
        lines.append("--- SHARE RATES BY CONTRARY NEIGHBORS ---")
        cat_mapping_broad = {
            'Left': 'Left (L / CL)',
            'Center-Left': 'Left (L / CL)',
            'Center': 'Neutral',
            'Center-Right': 'Right (R / CR)',
            'Right': 'Right (R / CR)'
        }
        
        df_cn = topic_df.copy()
        df_cn['broad_cat'] = df_cn['user_cat'].map(cat_mapping_broad)
        
        if 'contrary_neighbors' in df_cn.columns:
            cn_grouped = df_cn.groupby(['broad_cat', 'contrary_neighbors', 'action']).size().unstack(fill_value=0).reset_index()
            for col in ['seen', 'shared']:
                if col not in cn_grouped.columns: cn_grouped[col] = 0
            
            cn_grouped['share_rate'] = np.where(cn_grouped['seen'] > 0, cn_grouped['shared'] / cn_grouped['seen'] * 100, 0.0)
            
            for broad_c in ['Left (L / CL)', 'Neutral', 'Right (R / CR)']:
                cat_cn_df = cn_grouped[cn_grouped['broad_cat'] == broad_c].sort_values('contrary_neighbors')
                if not cat_cn_df.empty:
                    lines.append(f"\n{broad_c} Participants:")
                    for _, row in cat_cn_df.iterrows():
                        cn_count = int(row['contrary_neighbors'])
                        cn_rate = row['share_rate']
                        cn_seen = int(row['seen'])
                        cn_shared = int(row['shared'])
                        lines.append(f"  {cn_count} Contrary Neighbors: {cn_rate:>5.1f}% ({cn_shared} shared / {cn_seen} seen)")
        else:
            lines.append("  [!] 'contrary_neighbors' metric was not found in the event log.")
        lines.append("")

        # ---------------------------------------------------------
        # 5. ROUND-BY-ROUND BREAKDOWN
        # ---------------------------------------------------------
        src_metrics = stats_suite.calculate_sourcing_metrics(topic_df)
        def fmt(val): return f"{val:.2f}" if not np.isnan(val) else "N/A "

        for r in range(1, max_interactive_round + 1):
            lines.append(f"\n--- ROUND {r} ---")
            r_seen = topic_df[(topic_df['round'] == r) & (topic_df['action'] == 'seen')]
            r_share = topic_df[(topic_df['round'] == r) & (topic_df['action'] == 'shared')]
            
            lines.append(f"  Active Participants: {r_seen['user_id'].nunique()}")
            
            if src_metrics and r <= len(src_metrics['rounds']):
                r_idx = r - 1
                lines.append(f"  Sourcing Averages : {fmt(src_metrics['rep_avg'][r_idx])} Repeats | {fmt(src_metrics['pool_avg'][r_idx])} Pool | {fmt(src_metrics['back_avg'][r_idx])} Backlog")
                lines.append(f"  Expected Maximums : {fmt(src_metrics['rep_max'][r_idx])} Repeats | {fmt(src_metrics['pool_max'][r_idx])} Pool | {fmt(src_metrics['back_max'][r_idx])} Backlog")

            for uc in config.USER_CATS:
                u_seen = r_seen[r_seen['user_cat'] == uc]
                u_share = r_share[r_share['user_cat'] == uc]
                tot_cat_seen = len(u_seen)
                if tot_cat_seen > 0:
                    dist = [len(u_seen[u_seen['item_cat'] == ic]) / tot_cat_seen * 100 for ic in config.ITEM_CATS]
                    lines.append(f"  {uc:<12} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                    rates = [(len(u_share[u_share['item_cat'] == ic]) / len(u_seen[u_seen['item_cat'] == ic]) * 100) if len(u_seen[u_seen['item_cat'] == ic]) > 0 else 0 for ic in config.ITEM_CATS]
                    lines.append(f"  {uc:<12} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")

        # ---------------------------------------------------------
        # 6. CONSOLIDATED SHARE RATE MATRIX (ROUNDS 1-5)
        # ---------------------------------------------------------
        lines.append("\n" + "="*40)
        lines.append(f" SHARE RATE MATRIX (ROUNDS 1 TO {max_interactive_round})")
        lines.append("="*40)
        
        grouped = topic_df.groupby(['user_cat', 'item_cat', 'action']).size().unstack(fill_value=0).reset_index()
        
        for col in ['seen', 'shared']:
            if col not in grouped.columns:
                grouped[col] = 0
                
        grouped['share_rate'] = np.where(grouped['seen'] > 0, grouped['shared'] / grouped['seen'], 0.0)
        
        matrix = grouped.pivot(index='user_cat', columns='item_cat', values='share_rate').fillna(0)
        matrix = matrix.reindex(index=config.USER_CATS, columns=config.ITEM_CATS).fillna(0)
        
        lines.append("\n" + matrix.to_string(float_format=lambda x: f"{x:.1%}"))
        lines.append("\n")

        # ---------------------------------------------------------
        # 7. AGGREGATED SHARE RATE MATRIX (3 PARTICIPANT CATS)
        # ---------------------------------------------------------
        lines.append("="*40)
        lines.append(f" AGGREGATED PARTICIPANT MATRIX (3 CATEGORIES)")
        lines.append("="*40)
        
        cat_mapping = {
            'Left': 'Comb. Left',
            'Center-Left': 'Comb. Left',
            'Center': 'Center',
            'Center-Right': 'Comb. Right',
            'Right': 'Comb. Right'
        }
        
        agg_grouped = grouped.copy()
        agg_grouped['agg_user_cat'] = agg_grouped['user_cat'].map(cat_mapping)
        
        agg_counts = agg_grouped.groupby(['agg_user_cat', 'item_cat'])[['seen', 'shared']].sum().reset_index()
        agg_counts['share_rate'] = np.where(agg_counts['seen'] > 0, agg_counts['shared'] / agg_counts['seen'], 0.0)
        
        agg_matrix = agg_counts.pivot(index='agg_user_cat', columns='item_cat', values='share_rate').fillna(0)
        agg_matrix = agg_matrix.reindex(
            index=['Comb. Left', 'Center', 'Comb. Right'], 
            columns=config.ITEM_CATS
        ).fillna(0)
        
        lines.append("\n" + agg_matrix.to_string(float_format=lambda x: f"{x:.1%}"))
        lines.append("\n")

        safe_suffix = f"_{topic}" if is_dual else ""
        file_path = os.path.join(output_dir, f"Empirical_Comprehensive_Summary{safe_suffix}.txt")
        with open(file_path, 'w') as f: 
            f.write("\n".join(lines))
        print(f" -> Exported full details to Empirical_Comprehensive_Summary{safe_suffix}.txt")

def export_sweep_summary(sweep_aggregator, sweep_dir, k_val, regime_name, priority_str):
    """(Maintained to ensure simulation capabilities do not break)"""
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
                rnd_seen = [e for e in data['seen_events'] if e[3] == rnd]
                rnd_share = [e for e in data['share_events'] if e[3] == rnd]
                
                if src_metrics and rnd <= len(src_metrics['rounds']):
                    r_idx = rnd - 1
                    lines.append(f"       Sourcing Averages : {fmt(src_metrics['rep_avg'][r_idx])} Repeats | {fmt(src_metrics['pool_avg'][r_idx])} Pool | {fmt(src_metrics['back_avg'][r_idx])} Backlog")
                    lines.append(f"       Expected Maximums : {fmt(src_metrics['rep_max'][r_idx])} Repeats | {fmt(src_metrics['pool_max'][r_idx])} Pool | {fmt(src_metrics['back_max'][r_idx])} Backlog")
                
                user_stats = {uc: {'seen': {ic: 0 for ic in config.ITEM_CATS}, 'shared': {ic: 0 for ic in config.ITEM_CATS}} for uc in config.USER_CATS}
                for ev in rnd_seen: user_stats[config.get_bucket(ev[1], False)]['seen'][config.get_bucket(ev[2], True)] += 1
                for ev in rnd_share: user_stats[config.get_bucket(ev[1], False)]['shared'][config.get_bucket(ev[2], True)] += 1
                    
                for uc in config.USER_CATS:
                    tot_seen = sum(user_stats[uc]['seen'].values())
                    if tot_seen > 0:
                        dist = [user_stats[uc]['seen'][ic] / tot_seen * 100 for ic in config.ITEM_CATS]
                        rates = [(user_stats[uc]['shared'][ic] / user_stats[uc]['seen'][ic] * 100) if user_stats[uc]['seen'][ic] > 0 else 0.0 for ic in config.ITEM_CATS]
                        lines.append(f"       {uc:<12} Seen  : {dist[0]:>5.1f}% L | {dist[1]:>5.1f}% C-L | {dist[2]:>5.1f}% C | {dist[3]:>5.1f}% C-R | {dist[4]:>5.1f}% R")
                        lines.append(f"       {uc:<12} Share : {rates[0]:>5.1f}% L | {rates[1]:>5.1f}% C-L | {rates[2]:>5.1f}% C | {rates[3]:>5.1f}% C-R | {rates[4]:>5.1f}% R")
            lines.append("-" * 60)

    with open(os.path.join(sweep_dir, "Sweep_Summary.txt"), 'w') as f: f.write("\n".join(lines))
    print(f" -> Exported Sweep_Summary.txt")