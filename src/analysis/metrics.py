import numpy as np
import pandas as pd
import config

def build_matrix(df, round_filter=None):
    mat = np.zeros((5, 5))
    target_df = df if not round_filter else df[df['round'] == round_filter]
    
    user_cat_idx = {c: i for i, c in enumerate(config.USER_CATS)}
    item_cat_idx = {c: i for i, c in enumerate(config.ITEM_CATS)}
    
    for _, row in target_df.iterrows():
        u_idx = user_cat_idx.get(row['user_cat'])
        i_idx = item_cat_idx.get(row['item_cat'])
        if u_idx is not None and i_idx is not None:
            mat[u_idx, i_idx] += 1
    return mat

def calc_rate(mat, base_mat, mode='row'):
    with np.errstate(divide='ignore', invalid='ignore'):
        if mode == 'row':
            denom = base_mat.sum(axis=1, keepdims=True)
            rates = np.where(denom > 0, mat / denom, 0)
        else:
            denom = base_mat
            rates = np.where(denom > 0, mat / denom, 0)
            
    mask = np.zeros((5, 5), dtype=bool)
    if mode == 'row': mask = mask | np.broadcast_to(denom == 0, rates.shape) 
    else: mask = mask | (denom == 0)
        
    return np.ma.masked_where(mask, rates)

def calculate_sourcing_metrics(df):
    seen_df_base = df[df['action'] == 'seen'].copy()
    hid_df_base = df[df['action'] == 'hidden'].copy()
    if seen_df_base.empty: return {}
    
    first_seen = seen_df_base.groupby(['trial_id', 'item_id'])['round'].min().reset_index()
    first_events = pd.merge(first_seen, seen_df_base, on=['trial_id', 'item_id', 'round'])
    origins = first_events.groupby(['trial_id', 'item_id'])['chamber'].apply(lambda x: x.iloc[0] if x.nunique() == 1 else 'Both').reset_index(name='origin_chamber')
    
    seen_df_base = pd.merge(seen_df_base, origins, on=['trial_id', 'item_id'], how='left')
    hid_df_base = pd.merge(hid_df_base, origins, on=['trial_id', 'item_id'], how='left')
    
    seen_df_base['is_cross'] = (seen_df_base['origin_chamber'] != seen_df_base['chamber']) & (seen_df_base['origin_chamber'] != 'Both')
    rounds = sorted(seen_df_base['round'].unique())
    results = {}
    
    for scope in ['Global', 'Left', 'Right']:
        if 'chamber' not in seen_df_base.columns and scope != 'Global': continue
        
        scope_seen = seen_df_base if scope == 'Global' else seen_df_base[seen_df_base['chamber'] == scope]
        scope_hid = hid_df_base if scope == 'Global' else hid_df_base[hid_df_base['chamber'] == scope]
        if scope_seen.empty: continue
        
        metrics = {
            'rounds': rounds, 
            'rep_avg': [], 'rep_std': [], 'rep_max': [], 
            'pool_avg': [], 'pool_std': [], 'pool_max': [], 
            'back_avg': [], 'back_std': [], 'back_max': [],
            'hid_avg': [], 'hid_std': [], 'hid_max': [],
            'cross_avg': [], 'cross_std': [], 'cross_max': []
        }
        
        for r in rounds:
            r_df = scope_seen[scope_seen['round'] <= r]
            r_hid = scope_hid[scope_hid['round'] <= r]
            
            reps = r_df[r_df.duplicated(subset=['trial_id', 'user_id', 'item_id'], keep='first')]
            pools = r_df[(r_df['round'] > 1) & (r_df['source'] == 'pool')]
            backs = r_df[(r_df['round'] > 1) & (r_df['is_backlog'] == True)]
            crosses = r_df[r_df['is_cross'] == True]
            
            def get_aggs(filtered_df):
                counts_df = filtered_df.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
                base_users = scope_seen[['trial_id', 'user_id']].drop_duplicates()
                merged = pd.merge(base_users, counts_df, on=['trial_id', 'user_id'], how='left').fillna(0)
                return merged['count'].mean(), (merged['count'].std() if len(merged) > 1 else 0.0), (merged['count'].max() if len(merged) > 0 else 0.0)
                
            metrics['rep_avg'].append(get_aggs(reps)[0]); metrics['rep_std'].append(get_aggs(reps)[1]); metrics['rep_max'].append(get_aggs(reps)[2])
            metrics['cross_avg'].append(get_aggs(crosses)[0]); metrics['cross_std'].append(get_aggs(crosses)[1]); metrics['cross_max'].append(get_aggs(crosses)[2])
            
            if r > 1:
                metrics['pool_avg'].append(get_aggs(pools)[0]); metrics['pool_std'].append(get_aggs(pools)[1]); metrics['pool_max'].append(get_aggs(pools)[2])
                metrics['back_avg'].append(get_aggs(backs)[0]); metrics['back_std'].append(get_aggs(backs)[1]); metrics['back_max'].append(get_aggs(backs)[2])
                metrics['hid_avg'].append(get_aggs(r_hid)[0]); metrics['hid_std'].append(get_aggs(r_hid)[1]); metrics['hid_max'].append(get_aggs(r_hid)[2])
            else:
                for k in ['pool', 'back', 'hid']:
                    metrics[f'{k}_avg'].append(np.nan); metrics[f'{k}_std'].append(np.nan); metrics[f'{k}_max'].append(np.nan)
            
        results[scope] = metrics
    return results

def get_structural_outcomes(seen_events, n_nodes, chamber, metric, topic_filter=None, window_size=1):
    half_point = n_nodes / 2
    valid_nodes = [n for n in range(n_nodes) if n < half_point] if chamber == 'Left Chamber' else [n for n in range(n_nodes) if n >= half_point]
    
    exposures = {n: [] for n in valid_nodes}
    max_round = 1

    for ev in seen_events:
        if topic_filter is not None and len(ev) >= 8 and ev[7] != topic_filter:
            continue
        node = int(ev[0])
        if node in exposures:
            i_lean = float(ev[2])
            r = int(ev[3])
            max_round = max(max_round, r)
            exposures[node].append((r, i_lean))

    start_round = max(1, max_round - window_size + 1) if window_size is not None else 1
    outcomes = {}
    
    for n in valid_nodes:
        window_leans = [l for r, l in exposures[n] if r >= start_round]
        
        if not window_leans: 
            outcomes[n] = np.nan
        else:
            ords = [min(4, max(0, int(l * 5))) - 2 for l in window_leans]
            if metric == 'Feed_Slant':
                outcomes[n] = np.mean(ords)
            elif metric == 'Feed_Variance':
                outcomes[n] = np.var(ords, ddof=0)
            else:
                outcomes[n] = 0.0
                
    return outcomes

def get_polarization_gap(seen_events, n_nodes, topic_filter, window_size, node_opinions, base_topic='base_topic'):
    exposures = {n: [] for n in range(n_nodes)}
    max_round = 1

    for ev in seen_events:
        if topic_filter is not None and len(ev) >= 8 and ev[7] != topic_filter:
            continue
        n_id, i_lean, r = int(ev[0]), float(ev[2]), int(ev[3])
        max_round = max(max_round, r)
        exposures[n_id].append((r, i_lean))
        
    start_round = max(1, max_round - window_size + 1) if window_size is not None else 1
    
    anti_imm_slants, pro_imm_slants = [], []
    
    for n in range(n_nodes):
        window_leans = [l for r, l in exposures[n] if r >= start_round]
        if not window_leans: continue
        
        ords = [min(4, max(0, int(l * 5))) - 2 for l in window_leans]
        s_i = np.mean(ords)
        
        base_op = node_opinions[n].get(base_topic, node_opinions[n].get('opinion', 0.5))
        if base_op < 0.5:
            pro_imm_slants.append(s_i)
        else:
            anti_imm_slants.append(s_i)
            
    mean_anti = np.mean(anti_imm_slants) if anti_imm_slants else 0.0
    mean_pro = np.mean(pro_imm_slants) if pro_imm_slants else 0.0
    
    return mean_anti - mean_pro