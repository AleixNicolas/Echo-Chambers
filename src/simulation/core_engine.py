import random
import numpy as np
import copy
import config

def get_item_cat_index(lean_val):
    if lean_val < 0.2:
        return 0  # Left
    elif lean_val < 0.4:
        return 1  # Center-Left
    elif lean_val < 0.6:
        return 2  # Center
    elif lean_val < 0.8:
        return 3  # Center-Right
    return 4      # Right

def get_user_cat_index(cat_str):
    mapping = {'Left': 0, 'Center-Left': 1, 'Center': 2, 'Center-Right': 3, 'Right': 4}
    return mapping.get(cat_str, 2)

def sample_trial_demographics(base_nodes):
    node_profiles = {}
    total_nodes = len(base_nodes)
    
    # 1. Structural Noise: Assign Lurkers
    num_lurkers = int(total_nodes * getattr(config, 'SIM_LURKER_FRACTION', 0.0))
    lurker_nodes = set(random.sample(list(base_nodes.keys()), num_lurkers))
    
    for node_id, data in base_nodes.items():
        base_op = data.get('opinion', 0.5)
        
        # Determine Base Category
        if base_op < 0.4:
            base_cat = 'Left'
        elif base_op > 0.6:
            base_cat = 'Right'
        else:
            base_cat = 'Center'
            
        # Dynamically fetch the specific probabilities for this base category directly from config
        comp_matrix = getattr(config, 'EMPIRICAL_COMPOSITION_MATRIX', {})
        # Fallback to uniform if config is missing
        split_dict = comp_matrix.get(base_cat, {cat: 0.2 for cat in config.USER_CATS})
            
        # 2. Composition Logic (Toggleable via Config)
        if getattr(config, 'USE_COMPOSITION_NOISE', True):
            # NOISE ON: Roll the dice based on empirical probabilities from the matrix
            cats = list(split_dict.keys())
            probs = list(split_dict.values())
            # Normalize in case probabilities don't perfectly sum to 1.0 due to rounding
            probs = np.array(probs) / sum(probs)
            side_cat = np.random.choice(cats, p=probs)
        else:
            # NOISE OFF: Deterministic fallback. Check if the baseline JSON already 
            # assigned a static side topic; otherwise, pick the highest probability one.
            opinions_dict = data.get('opinions', {})
            side_topic_name = getattr(config, 'EMPIRICAL_SIDE_TOPIC', 'climate')
            if side_topic_name in opinions_dict:
                side_cat = opinions_dict[side_topic_name]
            else:
                side_cat = max(split_dict, key=split_dict.get)

        # Map category back to a numeric opinion for the engine
        side_op = config.SIDE_OPINION_VALUES.get(side_cat, 0.5)
        
        node_profiles[int(node_id)] = {
            'base_topic': base_op,
            'side_topic': side_op,
            'base_cat': base_cat,
            'side_cat': side_cat,
            'is_lurker': str(node_id) in lurker_nodes,
            'opinion': base_op
        }
        
    return node_profiles

def evaluate_share_decision(node_profile, topic, item_lean, treatment_key, chamber_key):
    # 1. Structural Noise: Lurkers
    if node_profile.get('is_lurker', False):
        return False
        
    # 2. Aleatoric Noise: Trembling Hand / Misclicks
    if getattr(config, 'USE_ALEATORIC_NOISE', False):
        if random.random() < getattr(config, 'ALEATORIC_ERROR_RATE', 0.0):
            return random.random() < 0.50  # 50/50 blind coin flip
            
    # 3. Matrix Math
    user_cat = node_profile['base_cat'] if topic == 'base_topic' else node_profile['side_cat']
    user_idx = get_user_cat_index(user_cat)
    item_idx = get_item_cat_index(item_lean)
    
    # Retrieve the correct nested matrix
    if treatment_key == 'integrated':
        matrix = config.SHARE_PROB_MATRICES[topic]['integrated']
    else:
        matrix = config.SHARE_PROB_MATRICES[topic].get(chamber_key, config.SHARE_PROB_MATRICES[topic]['Left'])
        
    base_prob = matrix[user_idx][item_idx]
    
    # Scale probability up to compensate for pure lurkers
    lurker_frac = getattr(config, 'SIM_LURKER_FRACTION', 0.0)
    compensated_prob = base_prob * (1.0 / (1.0 - lurker_frac)) if lurker_frac < 1.0 else 0
    
    # Strictly bound probability between 5% and 95% 
    final_prob = min(0.95, max(0.05, compensated_prob))
    
    return random.random() < final_prob

def run_single_trial(baseline_data, global_item_pool, regime_rules, treatment, priority_flag):
    nodes_data = baseline_data.get('nodes', {})
    network_adj = baseline_data.get('network', {})
    node_profiles = sample_trial_demographics(nodes_data)
    
    seen_events = []
    topics = getattr(config, 'SIM_TOPICS', ['base_topic', 'side_topic'])
    
    shared_history = {t: {n: set() for n in node_profiles} for t in topics}
    seen_history = {t: {n: set() for n in node_profiles} for t in topics}
    backlog = {t: {n: {} for n in node_profiles} for t in topics}
    last_round_shares = {t: {n: [] for n in node_profiles} for t in topics}
    
    # Identify chamber (Left nodes vs Right nodes) for appropriate matrix lookup in Segregated
    half_point = len(node_profiles) / 2
    
    for r in range(1, config.ROUNDS + 1):
        current_round_shares = {t: {n: [] for n in node_profiles} for t in topics}
        
        for topic in topics:
            pool = global_item_pool.get(topic, global_item_pool)
            pool_items = list(pool.keys())
            
            for n_id, profile in node_profiles.items():
                chamber_key = 'Left' if n_id < half_point else 'Right'
                feed_items = []
                
                if r == 1:
                    start_items = nodes_data.get(str(n_id), {}).get(f'{topic}_starting_items', [])
                    if not start_items:
                        start_items = nodes_data.get(str(n_id), {}).get('starting_items', [])
                    feed_items = [str(x) for x in start_items[:config.MAX_VISIBLE_ITEMS]]
                else:
                    neighbors = network_adj.get(str(n_id), [])
                    new_shares = {}
                    for nb in neighbors:
                        nb_int = int(nb)
                        for itm in last_round_shares[topic].get(nb_int, []):
                            if str(itm) not in shared_history[topic][n_id]:
                                new_shares[str(itm)] = new_shares.get(str(itm), 0) + 1
                                
                    new_never_seen = {k: v for k, v in new_shares.items() if str(k) not in seen_history[topic][n_id]}
                    back_never_seen = {k: v for k, v in backlog[topic][n_id].items() if str(k) not in seen_history[topic][n_id]}
                    new_seen = {k: v for k, v in new_shares.items() if str(k) in seen_history[topic][n_id] and str(k) not in shared_history[topic][n_id]}
                    back_seen = {k: v for k, v in backlog[topic][n_id].items() if str(k) in seen_history[topic][n_id] and str(k) not in shared_history[topic][n_id]}
                    
                    if priority_flag:
                        cascades = [new_never_seen, back_never_seen, new_seen, back_seen]
                    else:
                        combined_new = {**new_never_seen, **new_seen}
                        combined_back = {**back_never_seen, **back_seen}
                        cascades = [combined_new, combined_back]
                        
                    for c_dict in cascades:
                        c_keys = list(c_dict.keys())
                        c_weights = [c_dict[k] for k in c_keys]
                        while len(feed_items) < config.MAX_VISIBLE_ITEMS and c_keys:
                            choice = random.choices(c_keys, weights=c_weights, k=1)[0]
                            feed_items.append(choice)
                            idx = c_keys.index(choice)
                            c_keys.pop(idx)
                            c_weights.pop(idx)
                            
                    for k, v in new_shares.items():
                        clean_k = str(k)
                        if clean_k not in shared_history[topic][n_id]:
                            backlog[topic][n_id][clean_k] = backlog[topic][n_id].get(clean_k, 0) + v

                if len(feed_items) < config.MAX_VISIBLE_ITEMS:
                    avail_pool = [i for i in pool_items if str(i) not in feed_items and str(i) not in shared_history[topic][n_id]]
                    needed = config.MAX_VISIBLE_ITEMS - len(feed_items)
                    if avail_pool:
                        pool_never = [k for k in avail_pool if str(k) not in seen_history[topic][n_id]]
                        pool_seen = [k for k in avail_pool if str(k) in seen_history[topic][n_id]]
                        if pool_never:
                            drawn = random.sample(pool_never, min(needed, len(pool_never)))
                            feed_items.extend(drawn)
                            needed -= len(drawn)
                        if needed > 0 and pool_seen:
                            feed_items.extend(random.sample(pool_seen, min(needed, len(pool_seen))))
                            
                for itm_id in feed_items:
                    str_id = str(itm_id)
                    if str_id in backlog[topic][n_id]:
                        del backlog[topic][n_id][str_id]
                        
                    # FIX: Safely parse item leaning whether the JSON map uses a dictionary or a raw float
                    itm_data = pool.get(str_id, global_item_pool.get(str_id, 0.5))
                    if isinstance(itm_data, dict):
                        itm_lean = float(itm_data.get('leaning', 0.5))
                    else:
                        try:
                            itm_lean = float(itm_data)
                        except (ValueError, TypeError):
                            itm_lean = 0.5
                    
                    is_back = str_id in backlog[topic][n_id]
                    src = 'network' if r > 1 else 'seed'
                    
                    seen_events.append([n_id, str_id, itm_lean, r, 'seen', src, is_back, topic])
                    seen_history[topic][n_id].add(str_id)
                    
                    if evaluate_share_decision(profile, topic, itm_lean, treatment, chamber_key):
                        current_round_shares[topic][n_id].append(str_id)
                        shared_history[topic][n_id].add(str_id)
                        seen_events.append([n_id, str_id, itm_lean, r, 'shared', src, is_back, topic])
                        
        last_round_shares = current_round_shares

    return {
        'seen_events': seen_events,
        'node_opinions': node_profiles
    }

def run_batch(baseline_data, global_item_pool, regime_rules, n_trials, treatment, priority_flag):
    trials = []
    for _ in range(n_trials):
        t_res = run_single_trial(baseline_data, global_item_pool, regime_rules, treatment, priority_flag)
        trials.append(t_res)
    return {'trials': trials}