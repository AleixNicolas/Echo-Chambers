import pandas as pd
import json
import ast
import random
import re
from pathlib import Path
import config

def get_emp_item_bucket(lean_str):
    clean_str = str(lean_str).strip().title()
    if clean_str in ['Left']: return 'Left'
    if clean_str in ['Lean Left', 'Center Left', 'Center-Left']: return 'Center-Left'
    if clean_str in ['Lean Right', 'Center Right', 'Center-Right']: return 'Center-Right'
    if clean_str in ['Right']: return 'Right'
    return 'Center'

def get_emp_user_bucket(val):
    val = int(val)
    if val == 5: return 'Left'
    if val == 4: return 'Center-Left'
    if val == 3: return 'Center'
    if val == 2: return 'Center-Right'
    return 'Right'

def _safe_parse_list(val):
    if pd.isna(val) or str(val).strip() in ['', '[]']: return []
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list): return parsed
        except (ValueError, SyntaxError, TypeError):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list): return parsed
            except (ValueError, TypeError, json.JSONDecodeError): pass
    return []

def _safe_parse_dict(val):
    if pd.isna(val) or str(val).strip() in ['', '{}']: return {}
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict): return parsed
        except (ValueError, SyntaxError, TypeError):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict): return parsed
            except (ValueError, TypeError, json.JSONDecodeError): pass
    return {}

def get_col(base, r, topic, is_dual):
    return f'phase_2.{r}.player.{topic}_{base}' if is_dual else f'phase_2.{r}.player.{base}'

def _apply_canonical_metadata(df, network_data, topic, is_dual):
    treat_cols = [c for c in ['phase_2.1.player.network_treatment', 'participant.network_treatment'] if c in df.columns]
    df['canon_treatment'] = df[treat_cols].bfill(axis=1).iloc[:, 0].fillna('segregated').astype(str).str.strip() if treat_cols else 'segregated'

    node_cols = [f'phase_2.1.player.{topic}_node_id'] if is_dual else ['phase_2.1.player.node_id', 'participant.node_id']
    node_cols = [c for c in node_cols if c in df.columns]
    df['canon_node_id'] = pd.to_numeric(df[node_cols].bfill(axis=1).iloc[:, 0], errors='coerce').fillna(-1).astype(int).astype(str) if node_cols else "-1"

    op_cols = [f'phase_1.1.player.{topic}_opinion_2', f'phase_2.1.player.{topic}_opinion_2', f'participant.{topic}_opinion_2', f'{topic}_opinion_2'] if is_dual else [f'phase_1.1.player.{topic}_opinion_4', f'phase_2.1.player.{topic}_opinion_4', f'participant.{topic}_opinion_4', f'{topic}_opinion_4', 'phase_1.1.player.opinion_4', 'phase_2.1.player.opinion_4', 'participant.opinion_4', 'opinion_4']
    op_cols = [c for c in op_cols if c in df.columns]
    
    def _resolve_op(row):
        for c in op_cols:
            if pd.notna(row[c]):
                try:
                    val = max(1.0, min(5.0, float(row[c])))
                    return (6.0 - val) if topic in config.INVERT_OPINIONS_FOR else val
                except (ValueError, TypeError): pass
        
        cat_str = next((str(row[c]).strip().upper() for c in ['participant.assigned_category', 'phase_2.1.player.category', 'phase_1.1.player.category', 'participant.category'] if c in row and pd.notna(row[c]) and str(row[c]).strip()), None)
        if cat_str:
            if len(cat_str) >= 2 and cat_str in ['LL', 'LR', 'RL', 'RR']:
                char = cat_str[0] if topic == 'climate' else cat_str[1]
                return 5.0 if char == 'L' else (1.0 if char == 'R' else 3.0)
            elif not is_dual:
                return 5.0 if 'L' in cat_str else (1.0 if 'R' in cat_str else 3.0)
                
        treatment, node_id = row.get('canon_treatment', 'segregated'), row.get('canon_node_id', "-1")
        if node_id != "-1":
            try:
                op_val = network_data.get(f"{treatment}_baseline", {}).get('nodes', {}).get(node_id, {}).get('opinion')
                if op_val is not None:
                    return 5.0 if op_val < 0.2 else (4.0 if op_val < 0.4 else (3.0 if op_val < 0.6 else (2.0 if op_val < 0.8 else 1.0)))
            except (ValueError, TypeError): pass
        return 3.0

    df['canon_user_opinion'] = df.apply(_resolve_op, axis=1)
    return df

def append_final_hidden_feed(phase1_csv_path, phase2_csv_path, network_map_path, global_item_pool, topics=['climate']):
    phase2_csv_path = Path(phase2_csv_path)
    df = pd.read_csv(phase2_csv_path, low_memory=False)
    
    if phase1_csv_path and Path(phase1_csv_path).exists():
        df1 = pd.read_csv(phase1_csv_path, low_memory=False)
        if 'participant.label' in df1.columns and 'participant.label' in df.columns:
            cols_to_add = [c for c in df1.columns if c not in df.columns]
            if cols_to_add: df = pd.merge(df, df1[['participant.label'] + cols_to_add], on='participant.label', how='left')

    with open(network_map_path, 'r') as f: 
        network_data = json.load(f)
        
    is_dual = len(topics) > 1
    
    feed_cols = [c for c in df.columns if re.search(r'phase_2\.\d+\.player\.(?:.*_)?incoming_feed', c)]
    if not feed_cols: 
        return phase2_csv_path, config.ROUNDS
        
    last_round = max([int(re.search(r'phase_2\.(\d+)\.', c).group(1)) for c in feed_cols])
    next_round = last_round + 1
    
    for topic in topics:
        df = _apply_canonical_metadata(df, network_data, topic, is_dual)
        topic_pool = global_item_pool.get(topic, global_item_pool) if is_dual else global_item_pool
        topic_pool_ids = [str(k) for k in topic_pool.keys()] if isinstance(topic_pool, dict) else [str(k) for k in topic_pool]
        known_leanings = {}
        
        for r in range(1, last_round + 1):
            f_col = get_col('incoming_feed', r, topic, is_dual)
            if f_col in df.columns:
                for feed_raw in df[f_col].dropna():
                    for itm in _safe_parse_list(feed_raw):
                        if isinstance(itm, dict) and 'id' in itm and 'leaning' in itm: 
                            known_leanings[str(itm['id'])] = itm['leaning']

        next_round_feeds, node_to_uid, last_outgoing = {}, {}, {}
        for _, row in df.iterrows():
            node_id = row['canon_node_id']
            if node_id != "-1":
                treat_node_key = f"{row['canon_treatment']}_{node_id}"
                node_to_uid[treat_node_key] = str(row.get('participant.code', row.name))
                last_outgoing[treat_node_key] = _safe_parse_list(row.get(get_col('outgoing_shares', last_round, topic, is_dual), '[]'))

        for _, row in df.iterrows():
            node_id = row['canon_node_id']
            if node_id == "-1": continue

            uid = str(row.get('participant.code', row.name))
            treatment = row['canon_treatment']
            backlog = _safe_parse_dict(row.get(get_col('current_backlog', last_round, topic, is_dual), '{}'))
            
            seen_history = {str(item['id']) for r in range(1, last_round + 1) for item in _safe_parse_list(row.get(get_col('incoming_feed', r, topic, is_dual), '[]')) if isinstance(item, dict) and 'id' in item}
            shared_history = {str(item) for r in range(1, last_round + 1) for item in _safe_parse_list(row.get(get_col('outgoing_shares', r, topic, is_dual), '[]'))}
            neighbors = network_data.get(f"{treatment}_baseline", {}).get('network', {}).get(node_id, [])
            
            new_items = {}
            for nb in neighbors:
                for item in last_outgoing.get(f"{treatment}_{str(nb)}", []):
                    if str(item) not in shared_history: 
                        new_items[str(item)] = new_items.get(str(item), 0) + 1

            new_never_seen = {k: v for k, v in new_items.items() if str(k) not in seen_history}
            backlog_never_seen = {k: v for k, v in backlog.items() if str(k) not in seen_history}
            new_seen_not_shared = {k: v for k, v in new_items.items() if str(k) in seen_history and str(k) not in shared_history}
            backlog_seen_not_shared = {k: v for k, v in backlog.items() if str(k) in seen_history and str(k) not in shared_history}
            
            feed_item_ids = []
            for pool_dict in [new_never_seen, backlog_never_seen, new_seen_not_shared, backlog_seen_not_shared]:
                pool_keys, pool_weights = list(pool_dict.keys()), list(pool_dict.values())
                while len(feed_item_ids) < config.MAX_VISIBLE_ITEMS and pool_keys:
                    choice = random.choices(pool_keys, weights=pool_weights, k=1)[0]
                    feed_item_ids.append(choice)
                    idx = pool_keys.index(choice)
                    pool_keys.pop(idx)
                    pool_weights.pop(idx)
                    
            if len(feed_item_ids) < config.MAX_VISIBLE_ITEMS:
                available_pool = [i for i in topic_pool_ids if str(i) not in feed_item_ids and str(i) not in shared_history]
                if available_pool:
                    pool_never_seen = [k for k in available_pool if k not in seen_history]
                    pool_seen = [k for k in available_pool if k in seen_history]
                    
                    needed = config.MAX_VISIBLE_ITEMS - len(feed_item_ids)
                    if pool_never_seen:
                        drawn = random.sample(pool_never_seen, min(needed, len(pool_never_seen)))
                        feed_item_ids.extend(drawn)
                        needed -= len(drawn)
                    if needed > 0 and pool_seen:
                        feed_item_ids.extend(random.sample(pool_seen, min(needed, len(pool_seen))))

            next_round_feeds[uid] = json.dumps([{"id": item_id, "leaning": known_leanings.get(str(item_id), "Center")} for item_id in feed_item_ids])

        df[get_col('incoming_feed', next_round, topic, is_dual)] = df['participant.code'].map(next_round_feeds)
        
    # Make sure this return block is cleanly aligned at the root of the function!
    extended_csv_path = phase2_csv_path.parent / f"{phase2_csv_path.stem}_extended_R{next_round}.csv"
    df.to_csv(extended_csv_path, index=False)
    
    return extended_csv_path, next_round