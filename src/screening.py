import os
import json
import pandas as pd
import numpy as np
import config

def analyze_and_rank_participants(csv_path):
    """Evaluates telemetry to assign integrity and attentiveness scores."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    
    # Prefix mapping
    p = 'phase_1.1.player'
    consent_col = f'{p}.consent_participate'
    
    if consent_col in df.columns:
        df = df[df[consent_col].isin([1, 1.0, True, 'True'])].copy()
    else:
        print(f"  [!] Notice: Column '{consent_col}' missing. Skipping consent filter.")

    def calculate_trust_score(row):
        score = 100
        reasons = []

        def get_val(field, default=0):
            val = row.get(f'{p}.{field}')
            return float(val) if pd.notna(val) else default
            
        def get_str(field, default=""):
            val = row.get(f'{p}.{field}')
            return str(val) if pd.notna(val) else default

        width = get_val('window_width', 1920)
        is_mobile = width < 800

        # 1. HARD FAILS
        if get_val('is_honeypot_bot') == 1: 
            score -= 100
            reasons.append("Honeypot Triggered")
        if get_val('is_ai_bot') == 1: 
            score -= 100
            reasons.append("AI Phrase Detected")
            
        # 2. IMC Check
        if get_val('imc_failed') == 1:
            score -= 50
            reasons.append("Failed IMC Check")
        
        # 3. Copypaste & Unfocused DOM Injection 
        unfocused_typing = get_val('typing_while_unfocused')
        if unfocused_typing > 0:
            score -= 60
            reasons.append(f"Unfocused Typing ({int(unfocused_typing)}x)")
            
        jumps = get_val('large_text_jumps')
        active_time = get_val('typing_active_time')
        if jumps > 0 and active_time < 10:
            score -= 40
            reasons.append(f"Fast Copypaste ({int(jumps)}x in {active_time:.1f}s)")

        # 4. TYPING DYNAMICS (Desktop Only)
        if not is_mobile:
            iki = get_val('iki_variance')
            if 0 < iki < 500: 
                score -= 30
                reasons.append("Robotic Typing (Extreme Low Var)")
            
            cpm = get_val('typing_cpm')
            if cpm > 800:
                score -= 40
                reasons.append(f"High CPM ({cpm:.0f})")

        # 5. Mouse Trajectory Teleportation (Desktop Only)
        if not is_mobile:
            traj_str = get_str('mouse_trajectory_log', "")
            if traj_str.startswith('['):
                try:
                    trajectories = json.loads(traj_str)
                    teleports = sum(1 for action in trajectories if len(action.get('path', [])) < 2)
                    if teleports >= 2:
                        score -= 30
                        reasons.append(f"Mouse Teleporting ({teleports}x)")
                except json.JSONDecodeError:
                    pass

        # 6. FOCUS (Visibility State)
        blurs = get_val('dirty_click_count')
        if blurs > 3:
            penalty = int((blurs - 3) * 10)
            score -= penalty
            reasons.append(f"Hidden Tabs ({int(blurs)}x)")

        return max(0, score), f"Score: {max(0, score)} [{' | '.join(reasons) if reasons else 'Clean'}]"

    if df.empty: return df
    
    df[['trust_score', 'score_breakdown']] = df.apply(lambda r: calculate_trust_score(r), axis=1, result_type='expand')
    
    # Calculate initial opinion concern groupings
    op_cols = [f'{p}.opinion_{i}' for i in range(1, 5)]
    avail_op_cols = [c for c in op_cols if c in df.columns]
    
    if avail_op_cols:
        df['total_op'] = df[avail_op_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        df['category'] = df['total_op'].apply(lambda x: 'High_Concern' if x >= 16 else 'Low_Concern')
    else:
        df['category'] = 'Unknown'

    ranked = df[[f'{p}.prolific_id', 'category', 'trust_score', 'score_breakdown']].sort_values(by=['category', 'trust_score'], ascending=[True, False])
    
    out_path = os.path.join(config.EMPIRICAL_PROCESSED_DIR, 'ranked_results.csv')
    ranked.to_csv(out_path, index=False)
    
    return ranked