import json
import config
from src.empirical import prolific_admin, otree_parser
from src.analysis import event_logger, plotting, reporting

def run_empirical_analysis():
    print(f"=== STARTING EMPIRICAL PIPELINE: {config.CURRENT_EMPIRICAL_TRIAL_ID} ===")
    
    raw_csv_path = config.EMPIRICAL_RAW_DIR / 'all_apps_wide_2.csv'
    phase1_csv_path = config.EMPIRICAL_RAW_DIR / 'all_apps_wide_1.csv'
    network_map_path = config.EMPIRICAL_RAW_DIR / 'network_map.json'
    
    if not raw_csv_path.exists() or not network_map_path.exists():
        print(f"[!] Critical Error: Missing raw files in {config.EMPIRICAL_RAW_DIR}")
        return

    topics = [config.EMPIRICAL_BASE_TOPIC]
    if getattr(config, 'EMPIRICAL_SIDE_TOPIC', None):
        topics.append(config.EMPIRICAL_SIDE_TOPIC)

    reports_dir = config.EMPIRICAL_RESULTS_DIR / "reports"
    plots_dir = config.EMPIRICAL_RESULTS_DIR / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. ADMIN & ATTRITION CHECKS ---")
    prolific_admin.process_admin(
        raw_csv_path, 
        global_exclusions=config.EMPIRICAL_GLOBAL_EXCLUSIONS,
        out_dir=reports_dir
    )
    
    with open(network_map_path, 'r') as f:
        network_data = json.load(f)
        
    print("\n--- 2. PRE-PROCESSING & APPENDING FEED ---")
    extended_csv_path, new_total_rounds = otree_parser.append_final_hidden_feed(
        phase1_csv_path, raw_csv_path, network_map_path, network_data.get('global_item_pool', {}), topics
    )
    config.ROUNDS = new_total_rounds
    
    print("\n--- 3. DATA ADAPTER (UNIVERSAL LOGGING) ---")
    event_log_df = event_logger.build_empirical_event_log(
        phase1_csv_path, extended_csv_path, network_map_path, topics
    )
    
    print(f"\n--- 4. GENERATING PLOTS (1 to {config.ROUNDS}) ---")
    plotting.generate_suite(event_log_df, plots_dir, prefix="Empirical", topics=topics)
    plotting.generate_share_distributions(event_log_df, plots_dir, prefix="Empirical", topics=topics)
    plotting.generate_correlation_heatmaps(extended_csv_path, plots_dir, topics=topics, empirical=True)

    for topic in topics:
        plotting.plot_feed_slant_variance(event_log_df, plots_dir, topic, prefix="Empirical")
        
    # B4: Scan for new survey questions and plot them automatically if found
    plotting.plot_awareness_survey(extended_csv_path, event_log_df, plots_dir, topics)
    
    print("\n--- 5. EXPORTING REPORTS ---")
    reporting.export_screening_combinations(phase1_csv_path, reports_dir)
    # B3: Generate isolated H3 demographic comparison to remove confounder
    reporting.export_h3_report(event_log_df, reports_dir, topics)
    reporting.export_comprehensive_summary(
        event_log_df, reports_dir, is_empirical=True, 
        phase2_csv_path=extended_csv_path, topics=topics, 
        global_exclusions=config.EMPIRICAL_GLOBAL_EXCLUSIONS, 
        round_exclusions=config.EMPIRICAL_ROUND_EXCLUSIONS
    )
    
    print(f"\n=== EMPIRICAL ANALYSIS COMPLETE. Results saved to: {config.EMPIRICAL_RESULTS_DIR} ===")

    print("\n--- 5. EXPORTING REPORTS ---")
    reporting.export_screening_combinations(phase1_csv_path, reports_dir)
    
    # B3: Generate isolated H3 demographic comparison to remove confounder
    reporting.export_h3_report(event_log_df, reports_dir, topics)
    
    # B3 (Alternative): Generate Item-Level Logistic Regression
    reporting.export_h3_pooled_regression(event_log_df, reports_dir, topics)
    
    reporting.export_comprehensive_summary(
        event_log_df, reports_dir, is_empirical=True, 
        phase2_csv_path=extended_csv_path, topics=topics, 
        global_exclusions=config.EMPIRICAL_GLOBAL_EXCLUSIONS, 
        round_exclusions=config.EMPIRICAL_ROUND_EXCLUSIONS
    )
    
    print(f"\n=== EMPIRICAL ANALYSIS COMPLETE. Results saved to: {config.EMPIRICAL_RESULTS_DIR} ===")

if __name__ == "__main__":
    run_empirical_analysis()