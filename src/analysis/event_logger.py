import pandas as pd
import json
from collections import defaultdict
from src.empirical import otree_parser
import config

def get_bucket(val, is_item=False):
    cats = config.ITEM_CATS if is_item else config.USER_CATS
    return cats[min(4, max(0, int(val * 5)))]

def build_empirical_event_log(phase1_csv_path, phase2_csv_path, network_map_path, topics=['climate']):
    df = pd.read_csv(phase2_csv_path, low_memory=False)
    if phase1_csv_path and phase1_csv_path.exists():
        df1 = pd.read_csv(phase1_csv_path, low_memory=False)
        if 'participant.label' in df1.columns and 'participant.label' in df.columns:
            cols_to_add = [c for c in df1.columns if c not in df.columns]
            if cols_to_add: df = pd.merge(df, df1[['participant.label'] + cols_to_add], on='participant.label', how='left')
                
    if 'participant._is_bot' in df.columns: df = df[df['participant._is_bot'] == 0]
    with open(network_map_path, 'r') as f: network_data = json.load(f)
    
    is_dual = len(topics) > 1
    event_logs = []
    
    for topic in topics:
        df = otree_parser._apply_canonical_metadata(df, network_data, topic, is_dual)
        uid_to_node, node_to_uid, node_to_cat = {}, {}, defaultdict(dict)
        
        for _, row in df.iterrows():
            node_id = row['canon_node_id']
            if node_id != "-1":
                uid = str(row.get('participant.code', row.name)).upper()
                treatment = row['canon_treatment']
                uid_to_node[uid] = (treatment, node_id)
                if treatment not in node_to_uid: node_to_uid[treatment] = {}
                node_to_uid[treatment][node_id] = uid
                node_to_cat[treatment][str(node_id)] = otree_parser.get_emp_user_bucket(row['canon_user_opinion'])

        treatment_half_points = {treat_key: (max(int(n) for n in network_data.get(f"{treat_key}_baseline", {}).get('nodes', {}).keys()) + 1) / 2 if network_data.get(f"{treat_key}_baseline", {}).get('nodes', {}) else 10 for treat_key in node_to_uid.keys()}

        deliveries = defaultdict(lambda: defaultdict(set))
        for r in range(1, config.ROUNDS):
            for _, row in df.iterrows():
                uid = str(row.get('participant.code', row.name)).upper()
                if uid not in uid_to_node: continue
                
                treatment, node_id = uid_to_node[uid]
                adj_list = network_data.get(f"{treatment}_baseline", {}).get('network', {})
                if str(node_id) not in adj_list: continue
                
                s_col = otree_parser.get_col('outgoing_shares', r, topic, is_dual)
                for t_node in adj_list[str(node_id)]:
                    t_uid = node_to_uid.get(treatment, {}).get(str(t_node))
                    if t_uid: deliveries[t_uid][r+1].update([str(s) for s in otree_parser._safe_parse_list(row.get(s_col, '[]'))])

        for _, row in df.iterrows():
            node_id = row['canon_node_id']
            if node_id == "-1": continue
            
            uid = str(row.get('participant.code', row.name)).upper()
            op_raw = row['canon_user_opinion']
            user_cat = otree_parser.get_emp_user_bucket(op_raw)
            treatment = row['canon_treatment']
            trial_id = f"empirical_{treatment}"
            chamber = 'Left' if int(node_id) < treatment_half_points.get(treatment, 10) else 'Right'
            
            my_neighbors = network_data.get(f"{treatment}_baseline", {}).get('network', {}).get(str(node_id), [])
            contrary_count = sum(1 for nb in my_neighbors if ('Left' in user_cat and 'Right' in node_to_cat[treatment].get(str(nb), 'Center')) or ('Right' in user_cat and 'Left' in node_to_cat[treatment].get(str(nb), 'Center')))
            
            user_history, user_shared_history, parser_backlogs = set(), set(), set()
            
            for r in range(1, config.ROUNDS + 1):
                part_col = f'phase_2.{r}.player.participated_this_round'
                if part_col in df.columns and pd.notna(row.get(part_col)) and float(row.get(part_col)) == 0.0: continue 

                if r > 1: user_shared_history.update([str(s) for s in otree_parser._safe_parse_list(row.get(otree_parser.get_col('outgoing_shares', r-1, topic, is_dual), '[]'))])
                
                current_queue = parser_backlogs.union(deliveries[uid][r])
                for hid_id in {i for i in current_queue if i in user_shared_history}:
                    event_logs.append({'trial_id': trial_id, 'topic': topic, 'user_id': uid, 'treatment': treatment, 'chamber': chamber, 'op_raw': op_raw, 'user_cat': user_cat, 'contrary_neighbors': contrary_count, 'round': r, 'item_id': hid_id, 'item_cat': 'Center', 'action': 'hidden', 'source': 'neighbor', 'is_backlog': False})

                parser_backlogs = {i for i in parser_backlogs if i not in user_shared_history}
                
                feed, shares = otree_parser._safe_parse_list(row.get(otree_parser.get_col('incoming_feed', r, topic, is_dual), '[]')), otree_parser._safe_parse_list(row.get(otree_parser.get_col('outgoing_shares', r, topic, is_dual), '[]'))
                feed_ids = []
                for item in feed:
                    if not isinstance(item, dict) or not item.get('id'): continue
                    iid = str(item['id'])
                    feed_ids.append(iid)
                    item_cat = otree_parser.get_emp_item_bucket(item.get('leaning', 'Center'))
                    
                    src_type, is_b = ('start', False) if r == 1 else (('neighbor', False) if iid in deliveries[uid][r] else (('neighbor', True) if iid in parser_backlogs else ('pool', False)))

                    user_history.add(iid)
                    base_event = {'trial_id': trial_id, 'topic': topic, 'user_id': uid, 'treatment': treatment, 'chamber': chamber, 'op_raw': op_raw, 'user_cat': user_cat, 'contrary_neighbors': contrary_count, 'round': r, 'item_id': iid, 'item_cat': item_cat, 'source': src_type, 'is_backlog': is_b}
                    
                    event_logs.append({**base_event, 'action': 'seen'})
                    if iid in shares: event_logs.append({**base_event, 'action': 'shared'})

                parser_backlogs.update(deliveries[uid][r])
                parser_backlogs = {i for i in parser_backlogs if i not in feed_ids}

    base_columns = ['trial_id', 'topic', 'user_id', 'treatment', 'chamber', 'op_raw', 'user_cat', 'contrary_neighbors', 'round', 'item_id', 'item_cat', 'action', 'source', 'is_backlog']
    return pd.DataFrame(event_logs) if event_logs else pd.DataFrame(columns=base_columns)

def build_simulation_event_log(results_store):
    """
    Converts the raw simulation outputs into the standardized Data Contract format.
    Dynamically reads the user categories generated per trial to capture composition noise.
    """
    logs = []
    
    for struct in ['integrated', 'segregated']:
        if struct not in results_store: continue
        for t_idx, trial_data in enumerate(results_store[struct]['trials']):
            node_ops = trial_data.get('node_opinions', {})
            trial_id = f"sim_{struct}_{t_idx}"
            
            for ev in trial_data.get('seen_events', []):
                n_id = int(ev[0])
                itm_id = str(ev[1])
                itm_lean = float(ev[2])
                rnd = int(ev[3])
                action = str(ev[4])
                src = str(ev[5])
                is_back = bool(ev[6])
                topic = str(ev[7]) if len(ev) >= 8 else getattr(config, 'SIM_TOPICS', ['base_topic'])[0]
                
                # Dynamically pull the correct category drawn for THIS specific trial
                profile = node_ops.get(n_id, {})
                if topic == 'base_topic' or topic == config.EMPIRICAL_BASE_TOPIC:
                    u_cat = profile.get('base_cat', 'Center')
                else:
                    u_cat = profile.get('side_cat', 'Center')
                
                # Determine chamber physically (Nodes 0-19 are Left, 20-39 are Right)
                half_point = len(node_ops) / 2
                chamber = 'Left' if n_id < half_point else 'Right'
                
                logs.append({
                    'trial_id': trial_id,
                    'topic': topic,
                    'user_id': f"{trial_id}_{n_id}",
                    'treatment': struct,
                    'chamber': chamber,
                    'op_raw': float(profile.get('opinion', 0.5)),
                    'user_cat': u_cat,
                    'contrary_neighbors': 0, # Placeholder for structural topology metrics
                    'round': rnd,
                    'item_id': itm_id,
                    'item_cat': get_bucket(itm_lean, is_item=True),
                    'action': action,
                    'source': src,
                    'is_backlog': is_back
                })
                
    df = pd.DataFrame(logs)
    
    # Enforce data contract
    if not df.empty:
        for col, dtype in config.EVENT_LOG_SCHEMA.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
                
    return df