import random
import numpy as np
import config

def evaluate_share(node_op, item_lean, current_prob_matrix):
    if config.USE_ALEATORIC_NOISE and random.random() < config.ALEATORIC_ERROR_RATE:
        return random.choice([True, False])
        
    u_idx = min(4, max(0, int(node_op * 5)))
    i_idx = min(4, max(0, int(item_lean * 5)))
    return random.random() < current_prob_matrix[u_idx][i_idx]

def assign_side_topic_opinions(nodes):
    """
    Assigns side-topic opinions using either a probabilistic approach (simulating real-world variance)
    or the Largest Remainder method (guaranteeing exact demographic quotas per trial).
    """
    left_nodes = [int(i) for i, d in nodes.items() if d['opinion'] < 0.5]
    right_nodes = [int(i) for i, d in nodes.items() if d['opinion'] >= 0.5]
    
    cats = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
    method = getattr(config, 'SIDE_TOPIC_ASSIGNMENT_METHOD', 'probabilistic')

    def allocate_largest_remainder(node_list, split_dict):
        total = len(node_list)
        if total == 0: return []
        
        # 1. Calculate exact decimal seats and base integers
        exact = {cat: total * split_dict.get(cat, 0) for cat in cats}
        base = {cat: int(exact[cat]) for cat in cats}
        remainder = {cat: exact[cat] - base[cat] for cat in cats}
        
        # 2. Find how many leftover seats need to be assigned
        leftover = total - sum(base.values())
        
        # 3. Shuffle to handle exact remainder ties fairly, then sort by highest remainder
        shuffled_cats = cats[:]
        random.shuffle(shuffled_cats)
        sorted_cats = sorted(shuffled_cats, key=lambda c: remainder[c], reverse=True)
        
        # 4. Distribute leftover seats
        for cat in sorted_cats[:leftover]:
            base[cat] += 1
            
        # 5. Build the final array of values and shuffle who gets what
        ops = []
        for cat in cats:
            ops.extend([config.SIDE_OPINION_VALUES[cat]] * base[cat])
        random.shuffle(ops)
        return ops

    def allocate_probabilistic(node_list, split_dict):
        ops_map = [config.SIDE_OPINION_VALUES[cat] for cat in cats]
        weights = [split_dict.get(cat, 0) for cat in cats]
        return random.choices(ops_map, weights=weights, k=len(node_list))
        
    # Generate the allocations based on chosen method
    if method == 'largest_remainder':
        left_ops = allocate_largest_remainder(left_nodes, config.SIDE_TOPIC_SPLIT_LEFT_BASE)
        right_ops = allocate_largest_remainder(right_nodes, config.SIDE_TOPIC_SPLIT_RIGHT_BASE)
    else:
        left_ops = allocate_probabilistic(left_nodes, config.SIDE_TOPIC_SPLIT_LEFT_BASE)
        right_ops = allocate_probabilistic(right_nodes, config.SIDE_TOPIC_SPLIT_RIGHT_BASE)
        
    node_opinions = {}
    for idx, node in enumerate(left_nodes):
        node_opinions[node] = {
            'base_topic': nodes[str(node)]['opinion'],
            'side_topic': left_ops[idx]
        }
    for idx, node in enumerate(right_nodes):
        node_opinions[node] = {
            'base_topic': nodes[str(node)]['opinion'],
            'side_topic': right_ops[idx]
        }
        
    return node_opinions

def run_single_simulation(baseline, item_pool, regime, last_round_priority=False):
    network = baseline['network']
    nodes = baseline['nodes']
    
    # 1. Assign dual opinions for each node
    node_opinions = assign_side_topic_opinions(nodes)
    
    # 2. Duplicate item pools with isolated topic prefixes
    topic_item_pools = {
        topic: {f"{topic}_{k}": v for k, v in item_pool.items()}
        for topic in config.SIM_TOPICS
    }
    
    # 3. Initialize state per topic for each node
    node_data = {}
    for i in network.keys():
        n_int = int(i)
        node_data[n_int] = {}
        for topic in config.SIM_TOPICS:
            node_data[n_int][topic] = {
                'opinion': node_opinions[n_int][topic],
                'history': set(),
                'shared_history': set(),
                'starting_items': [f"{topic}_{itm}" for itm in nodes[str(i)]['starting_items']],
                'backlog': {},
                'incoming_queue': {}
            }
            
    hide_shared = regime.get('hide_shared', True)
    hide_ignored = regime.get('hide_ignored', False)
    
    seen_events, share_events = [], []
    topic_events = {t: {'seen_events': [], 'share_events': []} for t in config.SIM_TOPICS}
    
    for r in range(1, config.ROUNDS + 1):
        next_deliv_for_next_round = {
            topic: {int(i): {} for i in network.keys()}
            for topic in config.SIM_TOPICS
        }
        
        for node in node_data.keys():
            for topic in config.SIM_TOPICS:
                data = node_data[node][topic]
                t_pool = topic_item_pools[topic]
                run_matrix = config.SHARE_PROB_MATRICES[topic]
                
                feed_items, feed_source_map = [], {}
                
                if hide_shared:
                    data['backlog'] = {str(k): v for k, v in data['backlog'].items() if str(k) not in data['shared_history']}
                    new_items = {str(k): v for k, v in data['incoming_queue'].items() if str(k) not in data['shared_history']}
                else:
                    new_items = data['incoming_queue'].copy()
                
                if hide_ignored:
                    data['backlog'] = {str(k): v for k, v in data['backlog'].items() if str(k) not in data['history'] or str(k) in data['shared_history']}
                    new_items = {str(k): v for k, v in new_items.items() if str(k) not in data['history'] or str(k) in data['shared_history']}

                if r == 1:
                    for itm in data['starting_items']:
                        feed_items.append(str(itm))
                        feed_source_map[str(itm)] = ('start', False)
                else:
                    if last_round_priority:
                        pool_new = list(new_items.keys())
                        weights_new = [new_items[k] for k in pool_new]
                        while len(feed_items) < config.MAX_VISIBLE_ITEMS and pool_new:
                            choice = random.choices(pool_new, weights=weights_new, k=1)[0]
                            feed_items.append(str(choice))
                            feed_source_map[str(choice)] = ('neighbor', False)
                            idx = pool_new.index(choice)
                            pool_new.pop(idx)
                            weights_new.pop(idx)
                            
                        if len(feed_items) < config.MAX_VISIBLE_ITEMS:
                            pool_old = [str(k) for k in data['backlog'].keys() if str(k) not in feed_items]
                            weights_old = [data['backlog'][k] for k in pool_old]
                            while len(feed_items) < config.MAX_VISIBLE_ITEMS and pool_old:
                                choice = random.choices(pool_old, weights=weights_old, k=1)[0]
                                feed_items.append(str(choice))
                                feed_source_map[str(choice)] = ('neighbor', True)
                                idx = pool_old.index(choice)
                                pool_old.pop(idx)
                                weights_old.pop(idx)
                                
                        for k, v in new_items.items():
                            data['backlog'][str(k)] = data['backlog'].get(str(k), 0) + v
                            
                    else:
                        for k, v in new_items.items():
                            data['backlog'][str(k)] = data['backlog'].get(str(k), 0) + v
                        pool_combined = [str(k) for k in data['backlog'].keys() if str(k) not in feed_items]
                        weights_combined = [data['backlog'][k] for k in pool_combined]
                        while len(feed_items) < config.MAX_VISIBLE_ITEMS and pool_combined:
                            choice = random.choices(pool_combined, weights=weights_combined, k=1)[0]
                            feed_items.append(str(choice))
                            feed_source_map[str(choice)] = ('neighbor', True) 
                            idx = pool_combined.index(choice)
                            pool_combined.pop(idx)
                            weights_combined.pop(idx)

                if len(feed_items) < config.MAX_VISIBLE_ITEMS:
                    needed = config.MAX_VISIBLE_ITEMS - len(feed_items)
                    available_pool = [str(k) for k in t_pool.keys() if str(k) not in feed_items]
                    if hide_shared:
                        available_pool = [k for k in available_pool if k not in data['shared_history']]
                    if hide_ignored:
                        available_pool = [k for k in available_pool if k not in data['history']]
                    if available_pool:
                        for p in random.sample(available_pool, min(needed, len(available_pool))):
                            feed_items.append(p)
                            feed_source_map[p] = ('pool', False)

                for item_id in feed_items:
                    if str(item_id) in data['backlog']:
                        del data['backlog'][str(item_id)]

                for itm_id in feed_items:
                    data['history'].add(itm_id)
                    source, is_backlog = feed_source_map[itm_id]
                    item_lean = t_pool.get(str(itm_id), 0.5) 
                    
                    event_tuple = (node, data['opinion'], item_lean, r, itm_id, source, is_backlog, topic)
                    seen_events.append(event_tuple)
                    topic_events[topic]['seen_events'].append(event_tuple)
                    
                    if evaluate_share(data['opinion'], item_lean, run_matrix):
                        data['shared_history'].add(itm_id)
                        share_events.append(event_tuple)
                        topic_events[topic]['share_events'].append(event_tuple)
                        
                        # Deliver across the exact same structural network edges
                        for nb in network[str(node)]:
                            next_deliv_for_next_round[topic][int(nb)][itm_id] = (
                                next_deliv_for_next_round[topic][int(nb)].get(itm_id, 0) + 1
                            )

        for i in node_data.keys():
            for topic in config.SIM_TOPICS:
                node_data[i][topic]['incoming_queue'] = next_deliv_for_next_round[topic][i]

    return {
        "seen_events": seen_events, 
        "share_events": share_events, 
        "node_opinions": node_opinions,
        "topics": topic_events
    }

def run_batch(baseline, item_pool, regime, num_trials, last_round_priority=False):
    aggregated = {
        "seen_events": [], 
        "share_events": [], 
        "trials": [],
        "topics": {t: {"seen_events": [], "share_events": []} for t in config.SIM_TOPICS}
    }
    for _ in range(num_trials):
        res = run_single_simulation(baseline, item_pool, regime, last_round_priority)
        aggregated["seen_events"].extend(res["seen_events"])
        aggregated["share_events"].extend(res["share_events"])
        aggregated["trials"].append(res)
        for t in config.SIM_TOPICS:
            aggregated["topics"][t]["seen_events"].extend(res["topics"][t]["seen_events"])
            aggregated["topics"][t]["share_events"].extend(res["topics"][t]["share_events"])
            
    return aggregated