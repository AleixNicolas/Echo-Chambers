import pandas as pd
import json
import ast
import random
import re
import os
from collections import defaultdict
import config

def _safe_parse_list(val):
    if pd.isna(val) or str(val).strip() in ['', '[]']: return []
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list): return parsed
        except:
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list): return parsed
            except: pass
    return []

def _safe_parse_dict(val):
    if pd.isna(val) or str(val).strip() in ['', '{}']: return {}
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict): return parsed
        except:
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict): return parsed
            except: pass
    return {}

def get_col(base, r, topic, is_dual):
    return f'phase_2.{r}.player.{topic}_{base}' if is_dual else f'phase_2.{r}.player.{base}'

def get_treatment(row):
    for col in ['phase_2.1.player.network_treatment', 'participant.network_treatment']:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return 'segregated'

def get_node_id(row, topic, is_dual):
    col_dual = f'phase_2.1.player.{topic}_node_id'
    if is_dual and col_dual in row and pd.notna(row[col_dual]) and str(row[col_dual]).strip():
        return str(int(float(row[col_dual])))
    for col in ['phase_2.1.player.node_id', 'participant.node_id']:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(int(float(row[col])))
    return "-1"

def get_user_opinion(row, network_data, topic, is_dual):
    col_names = []
    if is_dual:
        col_names = [
            f'phase_1.1.player.{topic}_opinion_2',
            f'phase_2.1.player.{topic}_opinion_2',
            f'participant.{topic}_opinion_2', 
            f'{topic}_opinion_2'
        ]
    else:
        col_names = [
            f'phase_1.1.player.{topic}_opinion_4', f'phase_2.1.player.{topic}_opinion_4',
            f'participant.{topic}_opinion_4', f'{topic}_opinion_4',
            'phase_1.1.player.opinion_4', 'phase_2.1.player.opinion_4',
            'participant.opinion_4', 'opinion_4'
        ]
        
    for col in col_names:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                val = max(1.0, min(5.0, val))
                if topic in config.INVERT_OPINIONS_FOR:
                    return 6.0 - val
                else:
                    return val
            except:
                pass

    cat_str = None
    for col in ['participant.assigned_category', 'phase_2.1.player.category', 'phase_1.1.player.category', 'participant.category']:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            cat_str = str(row[col]).strip().upper()
            break
            
    if cat_str:
        if len(cat_str) >= 2 and cat_str in ['LL', 'LR', 'RL', 'RR']:
            char = cat_str[0] if topic == 'climate' else cat_str[1]
            if char == 'L': return 5.0
            if char == 'R': return 1.0
        elif not is_dual:
            if 'L' in cat_str: return 5.0
            if 'R' in cat_str: return 1.0

    treatment = get_treatment(row)
    node_id = get_node_id(row, topic, is_dual)
    
    if node_id != "-1":
        try:
            node_data = network_data.get(f"{treatment}_baseline", {}).get('nodes', {}).get(node_id, {})
            fallback_op = node_data.get('opinion')
            if fallback_op is not None:
                if fallback_op < 0.2: return 5.0
                elif fallback_op < 0.4: return 4.0
                elif fallback_op < 0.6: return 3.0
                elif fallback_op < 0.8: return 2.0
                else: return 1.0
        except: pass
    
    return 3.0

def append_final_hidden_feed(phase1_csv_path, phase2_csv_path, network_map_path, global_item_pool, topics=['climate']):
    df = pd.read_csv(phase2_csv_path, low_memory=False)
    
    if phase1_csv_path and os.path.exists(phase1_csv_path):
        df1 = pd.read_csv(phase1_csv_path, low_memory=False)
        if 'participant.label' in df1.columns and 'participant.label' in df.columns:
            cols_to_add = [c for c in df1.columns if c not in df.columns]
            if cols_to_add:
                df1_sub = df1[['participant.label'] + cols_to_add]
                df = pd.merge(df, df1_sub, on='participant.label', how='left')

    with open(network_map_path, 'r') as f: network_data = json.load(f)
    is_dual = len(topics) > 1
    
    feed_cols = [c for c in df.columns if re.search(r'phase_2\.\d+\.player\.(?:.*_)?incoming_feed', c)]
    if not feed_cols: return phase2_csv_path, config.ROUNDS
        
    last_round = max([int(re.search(r'phase_2\.(\d+)\.', c).group(1)) for c in feed_cols])
    next_round = last_round + 1
    
    for topic in topics:
        topic_pool = global_item_pool.get(topic, global_item_pool) if is_dual else global_item_pool
        topic_pool_ids = [str(k) for k in topic_pool.keys()] if isinstance(topic_pool, dict) else [str(k) for k in topic_pool]
        known_leanings = {}
        
        for r in range(1, last_round + 1):
            f_col = get_col('incoming_feed', r, topic, is_dual)
            if f_col in df.columns:
                for feed_raw in df[f_col].dropna():
                    try:
                        for itm in _safe_parse_list(feed_raw):
                            if 'id' in itm and 'leaning' in itm: known_leanings[str(itm['id'])] = itm['leaning']
                    except: pass

        next_round_feeds, node_to_uid, last_outgoing = {}, {}, {}
        for _, row in df.iterrows():
            uid = str(row.get('participant.code', row.name))
            treatment = get_treatment(row)
            node_id = get_node_id(row, topic, is_dual)
            
            if node_id != "-1":
                treat_node_key = f"{treatment}_{node_id}"
                node_to_uid[treat_node_key] = uid
                s_col = get_col('outgoing_shares', last_round, topic, is_dual)
                shares = _safe_parse_list(row.get(s_col, '[]'))
                last_outgoing[treat_node_key] = shares

        for _, row in df.iterrows():
            uid = str(row.get('participant.code', row.name))
            treatment = get_treatment(row)
            node_id = get_node_id(row, topic, is_dual)
            
            if node_id == "-1": continue

            b_col = get_col('current_backlog', last_round, topic, is_dual)
            backlog = _safe_parse_dict(row.get(b_col, '{}'))
                
            shared_history = set()
            for r in range(1, last_round + 1):
                s_col = get_col('outgoing_shares', r, topic, is_dual)
                for item in _safe_parse_list(row.get(s_col, '[]')): 
                    shared_history.add(str(item))

            baseline_key = f"{treatment}_baseline"
            neighbors = network_data.get(baseline_key, {}).get('network', {}).get(node_id, [])
            
            new_items = {}
            for nb in neighbors:
                for item in last_outgoing.get(f"{treatment}_{str(nb)}", []):
                    if str(item) not in shared_history: new_items[str(item)] = new_items.get(str(item), 0) + 1

            backlog = {str(k): v for k, v in backlog.items() if str(k) not in shared_history}
            feed_item_ids = []
            
            pool_new = list(new_items.keys())
            weights_new = [new_items[k] for k in pool_new]
            while len(feed_item_ids) < config.MAX_VISIBLE_ITEMS and pool_new:
                choice = random.choices(pool_new, weights=weights_new, k=1)[0]
                feed_item_ids.append(choice)
                idx = pool_new.index(choice)
                pool_new.pop(idx); weights_new.pop(idx)
                
            if len(feed_item_ids) < config.MAX_VISIBLE_ITEMS:
                pool_old = list(backlog.keys())
                weights_old = [backlog[k] for k in pool_old]
                while len(feed_item_ids) < config.MAX_VISIBLE_ITEMS and pool_old:
                    choice = random.choices(pool_old, weights=weights_old, k=1)[0]
                    feed_item_ids.append(choice)
                    idx = pool_old.index(choice)
                    pool_old.pop(idx); weights_old.pop(idx)
                    
            if len(feed_item_ids) < config.MAX_VISIBLE_ITEMS:
                needed = config.MAX_VISIBLE_ITEMS - len(feed_item_ids)
                available_pool = [i for i in topic_pool_ids if str(i) not in feed_item_ids and str(i) not in shared_history]
                if available_pool: feed_item_ids.extend([str(p) for p in random.sample(available_pool, min(needed, len(available_pool)))])

            formatted_feed = [{"id": item_id, "leaning": known_leanings.get(str(item_id), "Center")} for item_id in feed_item_ids]
            next_round_feeds[uid] = json.dumps(formatted_feed)

        n_col = get_col('incoming_feed', next_round, topic, is_dual)
        df[n_col] = df['participant.code'].map(next_round_feeds)
        
    extended_csv_path = phase2_csv_path.replace('.csv', f'_extended_R{next_round}.csv')
    df.to_csv(extended_csv_path, index=False)
    return extended_csv_path, next_round

def build_empirical_event_log(phase1_csv_path, phase2_csv_path, network_map_path, topics=['climate']):
    df = pd.read_csv(phase2_csv_path, low_memory=False)
    
    if phase1_csv_path and os.path.exists(phase1_csv_path):
        df1 = pd.read_csv(phase1_csv_path, low_memory=False)
        if 'participant.label' in df1.columns and 'participant.label' in df.columns:
            cols_to_add = [c for c in df1.columns if c not in df.columns]
            if cols_to_add:
                df1_sub = df1[['participant.label'] + cols_to_add]
                df = pd.merge(df, df1_sub, on='participant.label', how='left')
                
    if 'participant._is_bot' in df.columns: df = df[df['participant._is_bot'] == 0]
    
    with open(network_map_path, 'r') as f: network_data = json.load(f)
    is_dual = len(topics) > 1
    event_logs = []
    
    # Identify which index of the 2D category string represents the base topic
    base_idx = 0 if config.EMPIRICAL_BASE_TOPIC.lower() == 'climate' else 1
    
    for topic in topics:
        uid_to_node, node_to_uid = {}, {}
        node_to_cat = defaultdict(dict)
        
        for _, row in df.iterrows():
            node_id = get_node_id(row, topic, is_dual)
            if node_id != "-1":
                uid = str(row.get('participant.code', row.name)).upper()
                treatment = get_treatment(row)
                uid_to_node[uid] = (treatment, node_id)
                if treatment not in node_to_uid: node_to_uid[treatment] = {}
                node_to_uid[treatment][node_id] = uid
                op_raw = get_user_opinion(row, network_data, topic, is_dual)
                node_to_cat[treatment][str(node_id)] = config.get_emp_user_bucket(op_raw)

        deliveries = defaultdict(lambda: defaultdict(set))
        for r in range(1, config.ROUNDS):
            for _, row in df.iterrows():
                uid = str(row.get('participant.code', row.name)).upper()
                if uid not in uid_to_node: continue
                
                treatment, node_id = uid_to_node[uid]
                adj_list = network_data.get(f"{treatment}_baseline", {}).get('network', {})
                if str(node_id) not in adj_list: continue
                
                s_col = get_col('outgoing_shares', r, topic, is_dual)
                shares = _safe_parse_list(row.get(s_col, '[]'))
                for t_node in adj_list[str(node_id)]:
                    t_uid = node_to_uid.get(treatment, {}).get(str(t_node))
                    if t_uid: deliveries[t_uid][r+1].update([str(s) for s in shares])

        for _, row in df.iterrows():
            uid = str(row.get('participant.code', row.name)).upper()
            node_id = get_node_id(row, topic, is_dual)
            if node_id == "-1": continue
            
            op_raw = get_user_opinion(row, network_data, topic, is_dual)
            user_cat = config.get_emp_user_bucket(op_raw)
            treatment = get_treatment(row)
            trial_id = f"empirical_{treatment}"
            
            # Extract actual structural Chamber based on their Base Topic leaning
            chamber = 'Right'
            raw_cat_str = str(row.get('participant.assigned_category', row.get('phase_2.1.player.category', ''))).strip().upper()
            if len(raw_cat_str) >= 2:
                target_char = raw_cat_str[base_idx] if base_idx < len(raw_cat_str) else 'C'
                chamber = 'Left' if target_char == 'L' else 'Right'
            elif raw_cat_str.startswith('L') or 'LEFT' in raw_cat_str:
                chamber = 'Left'
            
            adj_list = network_data.get(f"{treatment}_baseline", {}).get('network', {})
            my_neighbors = adj_list.get(str(node_id), [])
            contrary_count = sum(1 for nb in my_neighbors if ('Left' in user_cat and 'Right' in node_to_cat[treatment].get(str(nb), 'Center')) or ('Right' in user_cat and 'Left' in node_to_cat[treatment].get(str(nb), 'Center')))
            
            user_history, user_shared_history, parser_backlogs = set(), set(), set()
            
            for r in range(1, config.ROUNDS + 1):
                if r > 1:
                    s_col_prev = get_col('outgoing_shares', r-1, topic, is_dual)
                    user_shared_history.update([str(s) for s in _safe_parse_list(row.get(s_col_prev, '[]'))])
                
                parser_backlogs = {i for i in parser_backlogs if i not in user_shared_history}
                
                f_col = get_col('incoming_feed', r, topic, is_dual)
                s_col = get_col('outgoing_shares', r, topic, is_dual)
                
                feed = _safe_parse_list(row.get(f_col, '[]'))
                shares = _safe_parse_list(row.get(s_col, '[]'))
                
                feed_ids = []
                for item in feed:
                    if not isinstance(item, dict) or not item.get('id'): continue
                    iid = str(item['id'])
                    feed_ids.append(iid)
                    item_cat = config.get_emp_item_bucket(item.get('leaning', 'Center'))
                    
                    src_type = 'pool'
                    is_b = False
                    if r == 1: src_type = 'start'
                    elif iid in deliveries[uid][r]: src_type = 'neighbor'
                    elif iid in parser_backlogs:
                        src_type = 'neighbor'
                        is_b = True

                    user_history.add(iid)
                    event_logs.append({
                        'trial_id': trial_id, 'topic': topic, 'user_id': uid, 'treatment': treatment, 
                        'chamber': chamber, 'op_raw': op_raw, 'user_cat': user_cat, 
                        'contrary_neighbors': contrary_count, 'round': r, 'item_id': iid, 
                        'item_cat': item_cat, 'action': 'seen', 'source': src_type, 'is_backlog': is_b
                    })
                    if iid in shares:
                        event_logs.append({
                            'trial_id': trial_id, 'topic': topic, 'user_id': uid, 'treatment': treatment, 
                            'chamber': chamber, 'op_raw': op_raw, 'user_cat': user_cat, 
                            'contrary_neighbors': contrary_count, 'round': r, 'item_id': iid, 
                            'item_cat': item_cat, 'action': 'shared', 'source': src_type, 'is_backlog': is_b
                        })

                parser_backlogs.update(deliveries[uid][r])
                parser_backlogs = {i for i in parser_backlogs if i not in feed_ids}

    base_columns = ['trial_id', 'topic', 'user_id', 'treatment', 'chamber', 'op_raw', 'user_cat', 'contrary_neighbors', 'round', 'item_id', 'item_cat', 'action', 'source', 'is_backlog']
    if not event_logs:
        return pd.DataFrame(columns=base_columns)
        
    return pd.DataFrame(event_logs)

def build_simulation_event_log(results_store):
    logs = []
    for struct in ['integrated', 'segregated']:
        if struct not in results_store: continue
        for t_idx, trial_data in enumerate(results_store[struct]['trials']):
            trial_id = f"{struct}_{t_idx}"
            
            if not trial_data['node_opinions']: continue
            max_node = max([int(n) for n in trial_data['node_opinions'].keys()])
            half_point = (max_node + 1) / 2
            
            for ev_type in ['seen_events', 'share_events']:
                action = 'seen' if ev_type == 'seen_events' else 'shared'
                for ev in trial_data[ev_type]:
                    if len(ev) >= 8:
                        node, u, i, rnd, itm_id, src, is_b, topic = ev
                    else:
                        node, u, i, rnd, itm_id, src, is_b = ev
                        topic = 'base_topic'
                        
                    # Structural chamber logic based on node position
                    chamber = 'Left' if int(node) < half_point else 'Right'
                        
                    logs.append({
                        'trial_id': trial_id, 'topic': topic, 'user_id': f"{trial_id}_{node}", 
                        'treatment': struct, 'chamber': chamber, 'op_raw': u, 
                        'user_cat': config.get_bucket(u, is_item=False), 'contrary_neighbors': 0,
                        'round': rnd, 'item_id': itm_id, 'item_cat': config.get_bucket(i, is_item=True), 
                        'action': action, 'source': src, 'is_backlog': is_b
                    })
    return pd.DataFrame(logs)