import os
import json
import config
from src import admin_checks, data_wrangle, plotting_engine, reporting_engine

GLOBAL_EXCLUSIONS = [] 
ROUND_EXCLUSIONS = {
    1: [],
    2: ['69e980c89da1ccaa91f5672a'],
    3: ['69e980c89da1ccaa91f5672a'],
    4: ['69e980c89da1ccaa91f5672a'],
    5: ['69e980c89da1ccaa91f5672a'],
    6: ['69e980c89da1ccaa91f5672a'] 
}

def run_empirical_analysis():
    print("=== STARTING EMPIRICAL DATA PIPELINE ===")
    
    raw_csv_path = os.path.join(config.EMPIRICAL_RAW_DIR, 'all_apps_wide_2.csv')
    phase1_csv_path = os.path.join(config.EMPIRICAL_RAW_DIR, 'all_apps_wide_1.csv')
    network_map_path = os.path.join(config.EMPIRICAL_RAW_DIR, 'network_map.json')
    
    if not os.path.exists(raw_csv_path) or not os.path.exists(network_map_path):
        print(f"[!] Critical Error: Missing files.")
        return

    topics = [config.EMPIRICAL_BASE_TOPIC, config.EMPIRICAL_SIDE_TOPIC]
    print(f"\n--- 0. PIPELINE TOPICS CONFIGURED ---")
    print(f"Base Anchor Topic: {topics[0]}")
    print(f"Secondary Topic  : {topics[1]}")

    print("\n--- 1. ADMIN & ATTRITION CHECKS ---")
    admin_checks.process_admin(raw_csv_path)
    
    with open(network_map_path, 'r') as f:
        network_data = json.load(f)
        
    print("\n--- 2. PRE-PROCESSING & APPENDING FEED ---")
    extended_csv_path, new_total_rounds = data_wrangle.append_final_hidden_feed(
        phase1_csv_path, raw_csv_path, network_map_path, network_data.get('global_item_pool', {}), topics
    )
    config.ROUNDS = new_total_rounds
    
    print("\n--- 3. DATA ADAPTER (UNIVERSAL LOGGING) ---")
    event_log_df = data_wrangle.build_empirical_event_log(
        phase1_csv_path, extended_csv_path, network_map_path, topics
    )
    
    print(f"\n--- 4. GENERATING PLOTS (1 to {config.ROUNDS}) ---")
    plotting_engine.generate_suite(event_log_df, config.EMPIRICAL_RESULTS_DIR, prefix="Empirical", topics=topics)
    plotting_engine.generate_share_distributions(event_log_df, config.EMPIRICAL_RESULTS_DIR, prefix="Empirical", topics=topics)
    plotting_engine.generate_correlation_heatmaps(extended_csv_path, config.EMPIRICAL_RESULTS_DIR, topics=topics)
    
    print("\n--- 5. EXPORTING REPORTS ---")
    reporting_engine.export_screening_combinations(phase1_csv_path, config.EMPIRICAL_RESULTS_DIR)
    reporting_engine.export_comprehensive_empirical_summary(
        event_log_df, config.EMPIRICAL_RESULTS_DIR, extended_csv_path, topics, GLOBAL_EXCLUSIONS, ROUND_EXCLUSIONS
    )
    
    print("\n=== EMPIRICAL ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    os.makedirs(config.EMPIRICAL_RAW_DIR, exist_ok=True)
    os.makedirs(config.EMPIRICAL_RESULTS_DIR, exist_ok=True)
    run_empirical_analysis()