import os
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import config
from src import stats_suite

def generate_suite(event_log_df, output_dir, prefix="", topics=['climate']):
    if event_log_df.empty or 'treatment' not in event_log_df.columns:
        print(f"[!] Warning: Event log is empty or missing data. Skipping plots for {prefix}.")
        return

    is_sim = "Sim" in prefix
    is_dual = len(topics) > 1
    
    for topic in topics:
        topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        
        for treatment in topic_df['treatment'].unique():
            t_df = topic_df[topic_df['treatment'] == treatment]
            safe_prefix = f"{prefix}_{treatment}_{topic}" if is_dual else f"{prefix}_{treatment}"
            
            plot_diet_categories(t_df, output_dir, safe_prefix)
            plot_diet_vs_amplification(t_df, output_dir, safe_prefix)
            plot_temporal_heatmaps(t_df, output_dir, safe_prefix)
            plot_feed_sources(t_df, output_dir, safe_prefix)
            plot_exposure_distributions(t_df, output_dir, safe_prefix, is_sim)
            plot_contrary_neighbors_effect(t_df, output_dir, safe_prefix)
            if is_dual and (topic != config.EMPIRICAL_BASE_TOPIC and topic != 'base_topic'):
                plot_cross_pressured_profiles(t_df, output_dir, safe_prefix)

def generate_share_distributions(event_log_df, output_dir, prefix="", topics=['climate']):
    if event_log_df.empty or 'treatment' not in event_log_df.columns: return
    is_sim = "Sim" in prefix
    is_dual = len(topics) > 1
    
    for topic in topics:
        topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        for treatment in topic_df['treatment'].unique():
            t_df = topic_df[topic_df['treatment'] == treatment]
            safe_prefix = f"{prefix}_{treatment}_{topic}" if is_dual else f"{prefix}_{treatment}"
            plot_share_distributions(t_df, output_dir, safe_prefix, is_sim)

def plot_share_distributions(df, output_dir, prefix, is_sim=False):
    if df.empty: return
    users_df = df[['trial_id', 'user_id', 'user_cat']].drop_duplicates()
    shares_df = df[df['action'] == 'shared']
    share_counts = shares_df.groupby(['trial_id', 'user_id']).size().reset_index(name='total_shares')
    
    merged = pd.merge(users_df, share_counts, on=['trial_id', 'user_id'], how='left')
    merged['total_shares'] = merged['total_shares'].fillna(0).astype(int)
    
    left_mask = merged['user_cat'].isin(['Left', 'Center-Left'])
    center_mask = merged['user_cat'] == 'Center'
    right_mask = merged['user_cat'].isin(['Right', 'Center-Right'])
    has_center = center_mask.sum() > 0
    
    def get_avg_dist(data):
        if data.empty: return pd.Series(dtype=float)
        counts = data.groupby(['trial_id', 'total_shares']).size().unstack(fill_value=0)
        return counts.mean(axis=0)

    l_dist = get_avg_dist(merged[left_mask])
    c_dist = get_avg_dist(merged[center_mask]) if has_center else pd.Series(dtype=float)
    r_dist = get_avg_dist(merged[right_mask])
    
    max_shares = int(max([d.index.max() if not d.empty else 0 for d in [l_dist, c_dist, r_dist]]))
    l_dist = l_dist.reindex(range(max_shares + 1), fill_value=0)
    c_dist = c_dist.reindex(range(max_shares + 1), fill_value=0) if has_center else c_dist
    r_dist = r_dist.reindex(range(max_shares + 1), fill_value=0)
    
    x_ticks = np.arange(max_shares + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if has_center:
        width = 0.25
        ax.bar(x_ticks - width, l_dist, width, color='#3498db', edgecolor='black', zorder=3, label='Combined Left (L/CL)')
        ax.bar(x_ticks, c_dist, width, color='#bdc3c7', edgecolor='black', zorder=3, label='Pure Center (C)')
        ax.bar(x_ticks + width, r_dist, width, color='#e74c3c', edgecolor='black', zorder=3, label='Combined Right (R/CR)')
    else:
        width = 0.35
        ax.bar(x_ticks - width/2, l_dist, width, color='#3498db', edgecolor='black', zorder=3, label='Combined Left (L/CL)')
        ax.bar(x_ticks + width/2, r_dist, width, color='#e74c3c', edgecolor='black', zorder=3, label='Combined Right (R/CR)')
    
    ax.set_xlabel("Total Items Shared")
    ax.set_ylabel("Avg Participants" if is_sim else "Number of Participants")
    ax.set_xticks(x_ticks)
    if not is_sim:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.legend(title="Participant Leaning", title_fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_06_Share_Dist_{prefix}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_exposure_distributions(df, output_dir, prefix, is_sim=False):
    seen_df = df[df['action'] == 'seen'].copy()
    if seen_df.empty: return
    if not is_sim: seen_df = seen_df[seen_df['round'] == seen_df['round'].max()]
        
    base = seen_df[['trial_id', 'round', 'user_id', 'user_cat']].drop_duplicates()
    item_counts = seen_df.groupby(['trial_id', 'round', 'user_id', 'item_cat']).size().unstack(fill_value=0).reset_index()
    for ic in config.ITEM_CATS:
        if ic not in item_counts.columns: item_counts[ic] = 0
            
    merged = pd.merge(base, item_counts, on=['trial_id', 'round', 'user_id'], how='left').fillna(0)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    left_mask = merged['user_cat'].isin(['Left', 'Center-Left'])
    center_mask = merged['user_cat'] == 'Center'
    right_mask = merged['user_cat'].isin(['Right', 'Center-Right'])
    has_center = center_mask.sum() > 0
    x_ticks = np.arange(5)
    
    def get_avg_dist(data, ic):
        if data.empty: return np.zeros(5)
        counts = data[ic].astype(int)
        grouped = data.assign(cnt=counts).groupby(['trial_id', 'round'])['cnt'].value_counts().unstack(fill_value=0)
        for c in range(5):
            if c not in grouped.columns: grouped[c] = 0
        return grouped[[0, 1, 2, 3, 4]].mean(axis=0).values

    for idx, ic in enumerate(config.ITEM_CATS):
        ax = axes[idx]
        l_dist = get_avg_dist(merged[left_mask], ic)
        c_dist = get_avg_dist(merged[center_mask], ic) if has_center else None
        r_dist = get_avg_dist(merged[right_mask], ic)
        
        if has_center:
            width = 0.25
            ax.bar(x_ticks - width, l_dist, width, color='#3498db', edgecolor='black', zorder=3, label='Left (L/CL)')
            ax.bar(x_ticks, c_dist, width, color='#bdc3c7', edgecolor='black', zorder=3, label='Center (C)')
            ax.bar(x_ticks + width, r_dist, width, color='#e74c3c', edgecolor='black', zorder=3, label='Right (R/CR)')
        else:
            width = 0.35
            ax.bar(x_ticks - width/2, l_dist, width, color='#3498db', edgecolor='black', zorder=3, label='Left (L/CL)')
            ax.bar(x_ticks + width/2, r_dist, width, color='#e74c3c', edgecolor='black', zorder=3, label='Right (R/CR)')
        
        ax.set_ylabel("Avg Participants" if is_sim else "Participants")
        ax.set_xlabel("Number of Items Seen")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.grid(True, axis='y', alpha=0.3, zorder=0)

    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=14, title="User Demographics", title_fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_05_Exposure_Dist_{prefix}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_diet_categories(df, output_dir, prefix):
    fig, ax = plt.subplots(figsize=(10, 6))
    item_colors = {'Left': '#3498db', 'Center': '#bdc3c7', 'Right': '#e74c3c'}
    x_indices = np.arange(config.ROUNDS)
    
    seen_df = df[df['action'] == 'seen'].copy()
    if seen_df.empty or 'chamber' not in seen_df.columns: return
    
    left_chamber_diets = {'Left': [], 'Center': [], 'Right': []}
    right_chamber_diets = {'Left': [], 'Center': [], 'Right': []}
    
    for r in range(1, config.ROUNDS + 1):
        r_df = seen_df[seen_df['round'] == r]
        
        def calc_props(target_chamber):
            sub = r_df[r_df['chamber'] == target_chamber]
            tot = len(sub) if len(sub) > 0 else 1
            return {
                'Left': len(sub[sub['item_cat'].isin(['Left', 'Center-Left'])]) / tot,
                'Center': len(sub[sub['item_cat'] == 'Center']) / tot,
                'Right': len(sub[sub['item_cat'].isin(['Right', 'Center-Right'])]) / tot
            }
            
        l_props = calc_props('Left')
        r_props = calc_props('Right')
        
        for k in ['Left', 'Center', 'Right']:
            left_chamber_diets[k].append(l_props[k])
            right_chamber_diets[k].append(r_props[k])

    bottom_l, bottom_r = np.zeros(config.ROUNDS), np.zeros(config.ROUNDS)
    width = 0.35
    
    for k in ['Left', 'Center', 'Right']:
        ax.bar(x_indices - width/2, left_chamber_diets[k], width, bottom=bottom_l, color=item_colors[k], edgecolor='white')
        bottom_l += np.array(left_chamber_diets[k])
        ax.bar(x_indices + width/2, right_chamber_diets[k], width, bottom=bottom_r, color=item_colors[k], edgecolor='black', hatch='//')
        bottom_r += np.array(right_chamber_diets[k])

    ax.set_ylabel("Proportion of Total Feed")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Round {r}\n(L | R)" for r in range(1, config.ROUNDS + 1)])
    
    legend_elements = [
        mpatches.Patch(color='#3498db', label='Left Items'), 
        mpatches.Patch(color='#bdc3c7', label='Center Items'), 
        mpatches.Patch(color='#e74c3c', label='Right Items'),
        mpatches.Patch(facecolor='gray', edgecolor='white', label='Left Chamber'), 
        mpatches.Patch(facecolor='gray', edgecolor='black', hatch='//', label='Right Chamber')
    ]
        
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize='small')
    plt.title(f"Diet by Structural Chamber\n({prefix})", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_01_Diet_{prefix}.pdf"), dpi=300)
    plt.close(fig)

def plot_diet_vs_amplification(df, output_dir, prefix):
    if df.empty or 'chamber' not in df.columns: return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    item_colors = {'Left': '#3498db', 'Center': '#bdc3c7', 'Right': '#e74c3c'}
    
    for idx, chamber in enumerate(['Left', 'Right']):
        ax = axes[idx]
        chamber_df = df[df['chamber'] == chamber]
        if chamber_df.empty: continue
            
        seen_tot = len(chamber_df[chamber_df['action'] == 'seen'])
        share_tot = len(chamber_df[chamber_df['action'] == 'shared'])
        
        seen_props = {k: 0.0 for k in ['Left', 'Center', 'Right']}
        share_props = {k: 0.0 for k in ['Left', 'Center', 'Right']}
        
        if seen_tot > 0:
            seen_props['Left'] = len(chamber_df[(chamber_df['action'] == 'seen') & (chamber_df['item_cat'].isin(['Left', 'Center-Left']))]) / seen_tot
            seen_props['Center'] = len(chamber_df[(chamber_df['action'] == 'seen') & (chamber_df['item_cat'] == 'Center')]) / seen_tot
            seen_props['Right'] = len(chamber_df[(chamber_df['action'] == 'seen') & (chamber_df['item_cat'].isin(['Right', 'Center-Right']))]) / seen_tot
            
        if share_tot > 0:
            share_props['Left'] = len(chamber_df[(chamber_df['action'] == 'shared') & (chamber_df['item_cat'].isin(['Left', 'Center-Left']))]) / share_tot
            share_props['Center'] = len(chamber_df[(chamber_df['action'] == 'shared') & (chamber_df['item_cat'] == 'Center')]) / share_tot
            share_props['Right'] = len(chamber_df[(chamber_df['action'] == 'shared') & (chamber_df['item_cat'].isin(['Right', 'Center-Right']))]) / share_tot

        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, list(seen_props.values()), width, label='Incoming Feed (Diet)', color=[item_colors[k] for k in seen_props.keys()], alpha=0.5, edgecolor='black')
        ax.bar(x + width/2, list(share_props.values()), width, label='Outgoing Shares (Amp)', color=[item_colors[k] for k in share_props.keys()], edgecolor='black')
        
        ax.set_title(f"{chamber} Chamber", fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Left Items', 'Center Items', 'Right Items'])
        ax.set_ylim(0, 1.0)
        if idx == 0: ax.set_ylabel("Proportion of Volume")
        ax.grid(axis='y', alpha=0.3)
        
    handles = [mpatches.Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Incoming Feed (Diet)'),
               mpatches.Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='Outgoing Shares (Amplification)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.suptitle(f"Diet vs. Amplification Gap\n({prefix})", y=1.15, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_08_Diet_vs_Amp_{prefix}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_feed_sources(df, output_dir, prefix):
    metrics = stats_suite.calculate_sourcing_metrics(df)
    if not metrics: return
    
    rounds = metrics['rounds']
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rounds, metrics['rep_avg'], marker='o', lw=2, color='#e74c3c', label='Average Repeats')
    ax.plot(rounds, metrics['pool_avg'], marker='s', lw=2, color='#3498db', label='Average Pool')
    ax.plot(rounds, metrics['back_avg'], marker='^', lw=2, color='#27ae60', label='Average Backlog')
    ax.plot(rounds, metrics['rep_max'], marker='x', lw=2, ls='--', color='#e74c3c', label='Expected Max Repeats')
    ax.plot(rounds, metrics['pool_max'], marker='x', lw=2, ls='--', color='#3498db', label='Expected Max Pool')
    ax.plot(rounds, metrics['back_max'], marker='x', lw=2, ls='--', color='#27ae60', label='Expected Max Backlog')
    
    ax.set_xlabel("Simulation Round"); ax.set_ylabel("Accumulated Items")
    ax.set_xticks(rounds); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_04_Feed_Sources_{prefix}.pdf"), dpi=300)
    plt.close(fig)

def plot_temporal_heatmaps(df, output_dir, prefix):
    cmap_base = plt.cm.Greens; cmap_base.set_bad(color='#d3d3d3')
    indiv_dir = os.path.join(output_dir, "individual_plots")
    os.makedirs(indiv_dir, exist_ok=True)
    
    for name, mode in [("02_InFeed_Comp", 'row'), ("03_Share_Rate", 'element'), ("03B_Norm_Share_Volume", 'norm_vol')]:
        fig_main, axes_main = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        for idx, chamber in enumerate(['Left', 'Right']):
            ax_main = axes_main[idx]
            if 'chamber' not in df.columns: continue
            chamber_df = df[df['chamber'] == chamber]
            
            # --- MAIN AGGREGATE DASHBOARD (ALL ROUNDS) ---
            if not chamber_df.empty:
                mat_seen = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'seen'], None)
                mat_share = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'shared'], None)
                mat_trg = mat_share if 'Share' in name else mat_seen
                
                if mode == 'norm_vol':
                    row_sums = mat_seen.sum(axis=1, keepdims=True)
                    v_rate = np.divide(mat_trg, row_sums, out=np.zeros_like(mat_trg, dtype=float), where=row_sums!=0)
                else:
                    v_rate = stats_suite.calc_rate(mat_trg, mat_seen, mode)
                    
                im = ax_main.imshow(v_rate, origin='upper', cmap=cmap_base, vmin=0, vmax=1.0)
                ax_main.set_title(f"{chamber} Chamber", fontweight='bold')
                ax_main.set_xticks(range(5)); ax_main.set_yticks(range(5))
                ax_main.set_xticklabels([l.replace(' ', '\n') for l in config.LEANING_ORDER])
                if idx == 0: ax_main.set_yticklabels([config.POS_LABELS[p] for p in [5, 4, 3, 2, 1]])
                if idx == 0: ax_main.set_ylabel("User Baseline Opinion", fontsize=12)
            
            # --- INDIVIDUAL ARCHIVE PLOTS (ROUND-BY-ROUND) ---
            if not chamber_df.empty:
                for r_target in [None] + list(range(1, config.ROUNDS + 1)):
                    mat_s = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'seen'], r_target)
                    mat_sh = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'shared'], r_target)
                    mat_t = mat_sh if 'Share' in name else mat_s
                    if mode == 'norm_vol':
                        r_sums = mat_s.sum(axis=1, keepdims=True)
                        v_r = np.divide(mat_t, r_sums, out=np.zeros_like(mat_t, dtype=float), where=r_sums!=0)
                    else:
                        v_r = stats_suite.calc_rate(mat_t, mat_s, mode)
                        
                    fig_indiv, ax_indiv = plt.subplots(figsize=(6, 6))
                    im_indiv = ax_indiv.imshow(v_r, origin='upper', cmap=cmap_base, vmin=0, vmax=1.0)
                    ax_indiv.set_xticks(range(5)); ax_indiv.set_yticks(range(5))
                    ax_indiv.set_xticklabels([l.replace(' ', '\n') for l in config.LEANING_ORDER])
                    ax_indiv.set_yticklabels([config.POS_LABELS[p] for p in [5, 4, 3, 2, 1]])
                    ax_indiv.set_title(f"{chamber} Chamber | {'All Rounds' if not r_target else f'Round {r_target}'}")
                    fig_indiv.colorbar(im_indiv, ax=ax_indiv, fraction=0.046, pad=0.04, label="Rate")
                    
                    r_str = f"Round_{r_target}" if r_target else "AllRounds"
                    fig_indiv.savefig(os.path.join(indiv_dir, f"Plot_{name}_{prefix}_{chamber}Chamber_{r_str}.pdf"), dpi=300, bbox_inches='tight')
                    plt.close(fig_indiv)

        if 'im' in locals():
            fig_main.colorbar(im, ax=axes_main.tolist(), fraction=0.015, pad=0.04, label="Rate")
        plt.suptitle(f"{name.replace('_', ' ')} (Aggregate)\n{prefix}", fontsize=14, y=1.05)
        plt.savefig(os.path.join(output_dir, f"Plot_{name}_{prefix}_SideBySide.pdf"), dpi=300, bbox_inches='tight')
        plt.close(fig_main)

def plot_contrary_neighbors_effect(df, output_dir, prefix):
    if df.empty or 'contrary_neighbors' not in df.columns: return
    
    # Calculate share rate per user
    user_stats = df.groupby(['user_id', 'contrary_neighbors', 'action']).size().unstack(fill_value=0).reset_index()
    for col in ['seen', 'shared']: 
        if col not in user_stats.columns: user_stats[col] = 0
    
    user_stats['share_rate'] = np.where(user_stats['seen'] > 0, user_stats['shared'] / user_stats['seen'], np.nan)
    user_stats = user_stats.dropna(subset=['share_rate'])
    if user_stats.empty: return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.pointplot(data=user_stats, x='contrary_neighbors', y='share_rate', errorbar=('ci', 95), capsize=.1, color='#2c3e50', ax=ax, markers='o', zorder=5)
    sns.stripplot(data=user_stats, x='contrary_neighbors', y='share_rate', color='#3498db', alpha=0.4, jitter=True, ax=ax, zorder=1)
    
    ax.set_title(f"Effect of Contrary Neighbors on Share Rate\n({prefix})", fontweight='bold')
    ax.set_xlabel("Number of Contrary Neighbors")
    ax.set_ylabel("Individual Share Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_09_Contrary_Neighbors_{prefix}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cross_pressured_profiles(df, output_dir, prefix):
    """Specifically plots behavior of the 4 2D quadrants (LL, LR, RL, RR) on the secondary topic."""
    if df.empty or 'chamber' not in df.columns: return
    
    # Identify the exact 2D quadrant based on Chamber (Base Topic) + User_Cat (Secondary Topic)
    def map_quadrant(row):
        base_lean = 'L' if row['chamber'] == 'Left' else 'R'
        side_lean = 'L' if 'Left' in row['user_cat'] else ('R' if 'Right' in row['user_cat'] else 'C')
        return f"{base_lean}{side_lean}"
        
    df['quadrant'] = df.apply(map_quadrant, axis=1)
    
    # We only want to plot the strict dual-profiles (ignore Center users here for clarity)
    target_quads = ['LL', 'LR', 'RL', 'RR']
    quad_df = df[df['quadrant'].isin(target_quads)]
    if quad_df.empty: return
    
    user_stats = quad_df.groupby(['user_id', 'quadrant', 'action']).size().unstack(fill_value=0).reset_index()
    for col in ['seen', 'shared']: 
        if col not in user_stats.columns: user_stats[col] = 0
    user_stats['share_rate'] = np.where(user_stats['seen'] > 0, user_stats['shared'] / user_stats['seen'], np.nan)
    user_stats = user_stats.dropna(subset=['share_rate'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    order = ['LL', 'LR', 'RL', 'RR']
    
    sns.barplot(data=user_stats, x='quadrant', y='share_rate', order=order, color='#bdc3c7', edgecolor='black', errorbar=None, ax=ax, alpha=0.7)
    sns.stripplot(data=user_stats, x='quadrant', y='share_rate', order=order, color='#c0392b', alpha=0.6, jitter=True, ax=ax)
    
    ax.set_title(f"Secondary Topic Sharing by Cross-Pressured Profiles\n({prefix})", fontweight='bold')
    ax.set_xlabel("Profile (Base Topic | Secondary Topic)")
    ax.set_ylabel("Individual Share Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Plot_10_Cross_Pressured_{prefix}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_correlation_heatmaps(phase2_csv, output_dir, topics=['climate'], empirical=False, phase1_csv=None):
    if phase1_csv is None:
        base_dir = os.path.dirname(phase2_csv)
        phase1_csv = os.path.join(base_dir, 'all_apps_wide_1.csv')
    
    df2 = pd.read_csv(phase2_csv, low_memory=False)
    
    if os.path.exists(phase1_csv):
        df1 = pd.read_csv(phase1_csv, low_memory=False)
        if 'participant.label' in df1.columns and 'participant.label' in df2.columns:
            df = pd.merge(df2, df1, on='participant.label', how='left', suffixes=('', '_p1'))
        elif 'participant.code' in df1.columns and 'participant.code' in df2.columns:
            df = pd.merge(df2, df1, on='participant.code', how='left', suffixes=('', '_p1'))
        else: df = df2
    else:
        df = df2
        
    if 'participant._is_bot' in df.columns: df = df[df['participant._is_bot'] == 0]
    is_dual = len(topics) > 1
    
    for topic in topics:
        # STRICT PHASE 1 ANCHORING
        op_cols = [c for c in df.columns if topic in c and ('phase_1' in c or 'baseline' in c)]
        op_cols = sorted(list(set(op_cols)))
        
        if not op_cols: continue
        
        data_rows = []
        leanings_5 = ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right']
        map_5_to_3 = {'Left': 'Left', 'Lean Left': 'Left', 'Center': 'Center', 'Lean Right': 'Right', 'Right': 'Right'}
        
        for _, row in df.iterrows():
            seen_5, shared_5 = {k: 0 for k in leanings_5}, {k: 0 for k in leanings_5}
            op_vals = {col: pd.to_numeric(row.get(col), errors='coerce') for col in op_cols}
            
            for r in range(1, config.ROUNDS + 1):
                r_cols = [c for c in df.columns if f'phase_2.{r}.player.' in c]
                if is_dual:
                    f_col = next((c for c in r_cols if 'incoming_feed' in c and topic in c), None)
                    s_col = next((c for c in r_cols if 'outgoing_shares' in c and topic in c), None)
                else:
                    f_col = next((c for c in r_cols if 'incoming_feed' in c), None)
                    s_col = next((c for c in r_cols if 'outgoing_shares' in c), None)
                if not f_col or not s_col: continue
                
                feed_raw, shares_raw = row.get(f_col, '[]'), row.get(s_col, '[]')
                feed = ast.literal_eval(feed_raw) if isinstance(feed_raw, str) and feed_raw.strip() != '' else []
                shares = ast.literal_eval(shares_raw) if isinstance(shares_raw, str) and shares_raw.strip() != '' else []
                
                for item in feed:
                    if isinstance(item, dict) and 'id' in item:
                        parts = str(item['id']).split('-')
                        lean = {'L': 'Left', 'LL': 'Lean Left', 'C': 'Center', 'LR': 'Lean Right', 'R': 'Right'}.get(parts[1], 'Center') if len(parts) >= 2 else 'Center'
                        seen_5[lean] += 1
                for iid in shares:
                    parts = str(iid).split('-')
                    lean = {'L': 'Left', 'LL': 'Lean Left', 'C': 'Center', 'LR': 'Lean Right', 'R': 'Right'}.get(parts[1], 'Center') if len(parts) >= 2 else 'Center'
                    shared_5[lean] += 1
                    
            row_data = op_vals.copy()
            seen_3, shared_3 = {'Left': 0, 'Center': 0, 'Right': 0}, {'Left': 0, 'Center': 0, 'Right': 0}
            
            for lean in leanings_5:
                mapped = map_5_to_3[lean]
                seen_3[mapped] += seen_5[lean]
                shared_3[mapped] += shared_5[lean]
                row_data[f'ShareRate_5_{lean}'] = shared_5[lean] / seen_5[lean] if seen_5[lean] > 0 else np.nan
            for lean in ['Left', 'Center', 'Right']:
                row_data[f'ShareRate_3_{lean}'] = shared_3[lean] / seen_3[lean] if seen_3[lean] > 0 else np.nan
            data_rows.append(row_data)

        plot_df = pd.DataFrame(data_rows).dropna(subset=op_cols, how='any')
        if plot_df.empty: continue
            
        col_labels = []
        for c in op_cols:
            if 'opinion_1' in c: col_labels.append('Op 1')
            elif 'opinion_2' in c: col_labels.append('Op 2')
            elif 'opinion_3' in c: col_labels.append('Op 3')
            elif 'opinion_4' in c: col_labels.append('Op 4')
            else: col_labels.append(c.split('.')[-1])
        
        def _plot_hm(leanings, pref, title, filename):
            corr_matrix = np.zeros((len(leanings), len(op_cols)))
            for r_idx, lean in enumerate(leanings):
                for c_idx, col in enumerate(op_cols):
                    tdf = plot_df.dropna(subset=[col, f'ShareRate_{pref}_{lean}'])
                    if not tdf.empty: corr_matrix[r_idx, c_idx] = tdf[col].corr(tdf[f'ShareRate_{pref}_{lean}'])
                    else: corr_matrix[r_idx, c_idx] = np.nan
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pd.DataFrame(corr_matrix, index=[f"{l} Items" for l in leanings], columns=col_labels), annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, ax=ax)
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close(fig)

        safe_suffix = f"_{topic}" if is_dual else ""
        _plot_hm(['Left', 'Center', 'Right'], '3', '3-Bucket Correlation', f'Plot_07A_Corr{safe_suffix}.pdf')
        
        # Only output the noisy 5-Bucket correlation if Empirical
        if empirical:
            _plot_hm(leanings_5, '5', '5-Bucket Correlation', f'Plot_07C_Corr{safe_suffix}.pdf')