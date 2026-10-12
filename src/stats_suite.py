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
            rep_counts = reps.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
            
            pools = r_df[(r_df['round'] > 1) & (r_df['source'] == 'pool')]
            pool_counts = pools.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
            
            backs = r_df[(r_df['round'] > 1) & (r_df['is_backlog'] == True)]
            back_counts = backs.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
            
            hid_counts = r_hid.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
            crosses = r_df[r_df['is_cross'] == True]
            cross_counts = crosses.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
            
            def get_aggs(counts_df):
                base_users = scope_seen[['trial_id', 'user_id']].drop_duplicates()
                merged = pd.merge(base_users, counts_df, on=['trial_id', 'user_id'], how='left').fillna(0)
                mean_val = merged['count'].mean()
                std_val = merged['count'].std() if len(merged) > 1 else 0.0
                max_val = merged['count'].max() if len(merged) > 0 else 0.0
                return mean_val, std_val, max_val
                
            rep_mean, rep_std, rep_max = get_aggs(rep_counts)
            cross_mean, cross_std, cross_max = get_aggs(cross_counts)
            
            if r > 1:
                pool_mean, pool_std, pool_max = get_aggs(pool_counts)
                back_mean, back_std, back_max = get_aggs(back_counts)
                hid_mean, hid_std, hid_max = get_aggs(hid_counts)
            else:
                pool_mean, pool_std, pool_max = np.nan, np.nan, np.nan
                back_mean, back_std, back_max = np.nan, np.nan, np.nan
                hid_mean, hid_std, hid_max = np.nan, np.nan, np.nan
                
            metrics['rep_avg'].append(rep_mean); metrics['rep_std'].append(rep_std); metrics['rep_max'].append(rep_max)
            metrics['pool_avg'].append(pool_mean); metrics['pool_std'].append(pool_std); metrics['pool_max'].append(pool_max)
            metrics['back_avg'].append(back_mean); metrics['back_std'].append(back_std); metrics['back_max'].append(back_max)
            metrics['hid_avg'].append(hid_mean); metrics['hid_std'].append(hid_std); metrics['hid_max'].append(hid_max)
            metrics['cross_avg'].append(cross_mean); metrics['cross_std'].append(cross_std); metrics['cross_max'].append(cross_max)
            
        results[scope] = metrics
    return results

def get_structural_outcomes(seen_events, n_nodes, chamber, metric, topic_filter=None, window_size=1):
    """
    Extracts metrics STRICTLY based on physical chamber position (Node ID).
    Uses window_size to evaluate only the final N rounds (default: 1 = final round only).
    """
    half_point = n_nodes / 2
    
    if chamber == 'Left Chamber':
        valid_nodes = [n for n in range(n_nodes) if n < half_point]
        is_left = True
    else:
        valid_nodes = [n for n in range(n_nodes) if n >= half_point]
        is_left = False

    exposures = {
        n: {'rounds': {r: {'congenial': 0, 'contrary': 0} for r in range(1, 20)}, 'leans': []}
        for n in valid_nodes
    }
    
    max_round = 1
    for ev in seen_events:
        if topic_filter is not None and len(ev) >= 8 and ev[7] != topic_filter:
            continue
            
        node = int(ev[0])
        if node in exposures:
            i_lean = float(ev[2])
            r = int(ev[3])
            max_round = max(max_round, r)
            
            exposures[node]['leans'].append((r, i_lean))
            
            is_cong = (i_lean < 0.4) if is_left else (i_lean > 0.6)
            is_cont = (i_lean > 0.6) if is_left else (i_lean < 0.4)
            
            if is_cong: exposures[node]['rounds'][r]['congenial'] += 1
            if is_cont: exposures[node]['rounds'][r]['contrary'] += 1

    start_round = max(1, max_round - window_size + 1) if window_size is not None else 1

    outcomes = {}
    for n in valid_nodes:
        window_leans = [l for r, l in exposures[n]['leans'] if r >= start_round]
        
        if metric == 'Feed_Slant':
            if not window_leans: outcomes[n] = np.nan
            else:
                ords = [min(4, max(0, int(l * 5))) - 2 for l in window_leans]
                outcomes[n] = np.mean(ords)
                
        elif metric == 'Feed_Variance':
            if not window_leans: outcomes[n] = np.nan
            else:
                ords = [min(4, max(0, int(l * 5))) - 2 for l in window_leans]
                outcomes[n] = np.var(ords, ddof=0)
                
        elif metric == 'Congenial_Count':
            outcomes[n] = sum(exposures[n]['rounds'][r]['congenial'] for r in range(start_round, max_round + 1))
            
        elif metric == 'Contrary_Count':
            outcomes[n] = sum(exposures[n]['rounds'][r]['contrary'] for r in range(start_round, max_round + 1))
            
        elif metric == 'Target_GT_2':
            outcomes[n] = sum(1 for r in range(start_round, max_round + 1) if exposures[n]['rounds'][r]['congenial'] > 2)
            
        elif metric == 'Opposite_EQ_0':
            outcomes[n] = sum(1 for r in range(start_round, max_round + 1) if exposures[n]['rounds'][r]['contrary'] == 0)
            
        else:
            outcomes[n] = 0.0
            
    return outcomes