import os
import json
import pandas as pd
import config
from src import admin_checks, data_wrangle, plotting_engine, reporting_engine

# =======================================================
# --- MANUAL EXCLUSIONS ---
# Add the participant.label (Prolific ID) to remove them 
# from the valid active pool for specific rounds (or globally).
# They will still count towards Total Enrolled, but if they 
# miss >1 round (leaving them with <4 valid rounds in 1-5), 
# they become a drop-out.
# =======================================================
GLOBAL_EXCLUSIONS = [] # Excludes from ALL rounds

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
    
    # =======================================================
    # --- PATH DIAGNOSTICS ---
    # =======================================================
    raw_csv_path = os.path.join(config.EMPIRICAL_RAW_DIR, 'all_apps_wide_2.csv')
    phase1_csv_path = os.path.join(config.EMPIRICAL_RAW_DIR, 'all_apps_wide_1.csv')
    network_map_path = os.path.join(config.EMPIRICAL_RAW_DIR, 'network_map.json')
    
    csv_exists = os.path.exists(raw_csv_path)
    net_exists = os.path.exists(network_map_path)
    
    if not csv_exists or not net_exists:
        print(f"[!] Critical Error: Missing files.")
        print(f"CSV target: {os.path.abspath(raw_csv_path)} (Found: {csv_exists})")
        print(f"JSON target: {os.path.abspath(network_map_path)} (Found: {net_exists})")
        return

    # =======================================================
    # --- AUTO-DETECT SINGLE VS DUAL TOPIC ---
    # =======================================================
    print("\n--- 0. AUTO-DETECT MODE ---")
    topics = ['climate'] # Default fallback
    is_dual_topic = False
    
    try:
        sample_df = pd.read_csv(raw_csv_path, nrows=0)
        cols_str = " ".join(sample_df.columns)
        
        if 'climate_incoming_feed' in cols_str or 'climate_outgoing_shares' in cols_str:
            is_dual_topic = True
            topics = ['climate', 'imm']
            print("[*] SUCCESS: Auto-detected DUAL-TOPIC mode from CSV headers.")
        else:
            print("[*] SUCCESS: Auto-detected SINGLE-TOPIC mode from CSV headers.")
    except Exception as e:
        print(f"[?] Could not read CSV headers for auto-detection: {e}. Defaulting to single-topic.")

    print("\n--- 1. ADMIN & ATTRITION CHECKS ---")
    admin_checks.process_admin(raw_csv_path)
    
    with open(network_map_path, 'r') as f:
        network_data = json.load(f)
        
    print("\n--- 2. PRE-PROCESSING & APPENDING FEED ---")
    extended_csv_path, new_total_rounds = data_wrangle.append_final_hidden_feed(
        phase1_csv_path=phase1_csv_path,
        phase2_csv_path=raw_csv_path,
        network_map_path=network_map_path,
        global_item_pool=network_data.get('global_item_pool', {}),
        topics=topics
    )
    config.ROUNDS = new_total_rounds
    
    print("\n--- 3. DATA ADAPTER (UNIVERSAL LOGGING) ---")
    event_log_df = data_wrangle.build_empirical_event_log(
        phase1_csv_path=phase1_csv_path,
        phase2_csv_path=extended_csv_path,
        network_map_path=network_map_path,
        topics=topics
    )
    
    print(f"\n--- 4. GENERATING PLOTS (1 to {config.ROUNDS}) ---")
    plotting_engine.generate_suite(
        event_log_df, 
        config.EMPIRICAL_RESULTS_DIR, 
        prefix="Empirical", 
        topics=topics
    )
    plotting_engine.generate_share_distributions(
        event_log_df,
        config.EMPIRICAL_RESULTS_DIR,
        prefix="Empirical",
        topics=topics
    )
    plotting_engine.generate_correlation_heatmaps(
        extended_csv_path, 
        config.EMPIRICAL_RESULTS_DIR, 
        topics=topics
    )
    
    print("\n--- 5. EXPORTING REPORTS ---")
    reporting_engine.export_screening_combinations(
        phase1_csv_path=phase1_csv_path,
        output_dir=config.EMPIRICAL_RESULTS_DIR
    )
    reporting_engine.export_comprehensive_empirical_summary(
        event_log_df, 
        config.EMPIRICAL_RESULTS_DIR,
        phase2_csv_path=extended_csv_path,
        topics=topics,
        global_exclusions=GLOBAL_EXCLUSIONS,
        round_exclusions=ROUND_EXCLUSIONS
    )
    
    print("\n=== EMPIRICAL ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    os.makedirs(config.EMPIRICAL_RAW_DIR, exist_ok=True)
    os.makedirs(config.EMPIRICAL_RESULTS_DIR, exist_ok=True)
    run_empirical_analysis()