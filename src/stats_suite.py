import numpy as np
import pandas as pd
import config

def build_matrix(df, round_filter=None):
    """Converts a Universal DataFrame action list into a 5x5 numpy matrix."""
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
    """Calculates transformation rates safely (prevents division by zero)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if mode == 'row':
            denom = base_mat.sum(axis=1, keepdims=True)
            rates = np.where(denom > 0, mat / denom, 0)
        else:
            denom = base_mat
            rates = np.where(denom > 0, mat / denom, 0)
            
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, :] = True
    
    if mode == 'row': mask = mask | np.broadcast_to(denom == 0, rates.shape) 
    else: mask = mask | (denom == 0)
        
    return np.ma.masked_where(mask, rates)

def calculate_sourcing_metrics(df):
    """Calculates Sourcing Metrics across trials."""
    seen_df = df[df['action'] == 'seen']
    if seen_df.empty: return {}
    
    rounds = sorted(seen_df['round'].unique())
    metrics = {
        'rounds': rounds, 
        'rep_avg': [], 'rep_max': [], 
        'pool_avg': [], 'pool_max': [], 
        'back_avg': [], 'back_max': []
    }
    
    for r in rounds:
        r_df = seen_df[seen_df['round'] <= r]
        
        reps = r_df[r_df.duplicated(subset=['user_id', 'item_id'], keep='first')]
        rep_counts = reps.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
        
        pools = r_df[(r_df['round'] > 1) & (r_df['source'] == 'pool')]
        pool_counts = pools.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
        
        backs = r_df[(r_df['round'] > 1) & (r_df['is_backlog'] == True)]
        back_counts = backs.groupby(['trial_id', 'user_id']).size().reset_index(name='count')
        
        def get_aggs(counts_df):
            base_users = r_df[['trial_id', 'user_id']].drop_duplicates()
            merged = pd.merge(base_users, counts_df, on=['trial_id', 'user_id'], how='left').fillna(0)
            trial_aggs = merged.groupby('trial_id')['count'].agg(['mean', 'max'])
            return trial_aggs['mean'].mean(), trial_aggs['max'].mean()
            
        rep_mean, rep_max = get_aggs(rep_counts)
        if r > 1:
            pool_mean, pool_max = get_aggs(pool_counts)
            back_mean, back_max = get_aggs(back_counts)
        else:
            pool_mean, pool_max = np.nan, np.nan
            back_mean, back_max = np.nan, np.nan
            
        metrics['rep_avg'].append(rep_mean)
        metrics['rep_max'].append(rep_max)
        metrics['pool_avg'].append(pool_mean)
        metrics['pool_max'].append(pool_max)
        metrics['back_avg'].append(back_mean)
        metrics['back_max'].append(back_max)
        
    return metrics

def extract_node_outcomes(seen_events, nodes_data, cat, metric, window_size=None, topic_filter=None):
    """
    Extracts polarization outcomes.
    nodes_data can be {node_str: {'opinion': val}} or {node_int: val}.
    topic_filter allows isolating events from a specific topic ('base_topic' or 'side_topic').
    """
    valid_nodes = []
    for n_key, info in nodes_data.items():
        op = info['opinion'] if (isinstance(info, dict) and 'opinion' in info) else float(info)
        if (cat == 'Left Users' and op < 0.5) or (cat == 'Right Users' and op > 0.5):
            valid_nodes.append(int(n_key))

    exposures = {
        n: {'congenial': 0, 'contrary': 0, 'rounds': {r: {'congenial': 0, 'contrary': 0} for r in range(1, 20)}}
        for n in valid_nodes
    }
    max_round = 1
    
    for event in seen_events:
        node = int(event[0])
        i_lean = float(event[2])
        r = int(event[3])
        
        # Topic filter verification
        if topic_filter is not None and len(event) >= 8:
            if event[7] != topic_filter:
                continue
                
        if node not in exposures: continue
        
        max_round = max(max_round, r)
        
        is_congenial = (cat == 'Left Users' and i_lean < 0.4) or (cat == 'Right Users' and i_lean > 0.6)
        is_contrary = (cat == 'Left Users' and i_lean > 0.6) or (cat == 'Right Users' and i_lean < 0.4)
            
        if is_congenial:
            exposures[node]['congenial'] += 1
            exposures[node]['rounds'][r]['congenial'] += 1
        if is_contrary:
            exposures[node]['contrary'] += 1
            exposures[node]['rounds'][r]['contrary'] += 1

    if window_size is not None and window_size > 0:
        start_round = max(1, max_round - window_size + 1)
    else:
        start_round = 1 

    outcomes = {}
    for n in valid_nodes:
        if metric == 'Congenial_Count': 
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