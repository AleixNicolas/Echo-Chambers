import os

# ==========================================
# 1. PIPELINE MODE & MASTER SWITCHES
# ==========================================
# Determines which topic forms the structural backbone of the network.
# Used by the data wrangler to map 2D profiles into physical Left/Right chambers.
EMPIRICAL_BASE_TOPIC = "imm"  
EMPIRICAL_SIDE_TOPIC = "climate"

# Topics that require their 1-5 scale to be inverted (e.g. 5 becomes 1) 
# so that 5 ALWAYS means "Left" and 1 ALWAYS means "Right".
INVERT_OPINIONS_FOR = ['imm', 'immigration']

# ==========================================
# 2. PATH CONFIGURATION & FOLDER STRUCTURE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---> Change this string when you run a new human cohort! <---
CURRENT_EMPIRICAL_TRIAL_ID = "Trial_2026_09_07"

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BASELINES_DIR = os.path.join(DATA_DIR, "baselines")

EMPIRICAL_DATA_DIR = os.path.join(DATA_DIR, "empirical", CURRENT_EMPIRICAL_TRIAL_ID)
EMPIRICAL_RAW_DIR = os.path.join(EMPIRICAL_DATA_DIR, "raw")
EMPIRICAL_PROCESSED_DIR = os.path.join(EMPIRICAL_DATA_DIR, "processed")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")
EMPIRICAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "Empirical", CURRENT_EMPIRICAL_TRIAL_ID)
SIMULATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "Simulations_dual")

for d in [BASELINES_DIR, EMPIRICAL_RAW_DIR, EMPIRICAL_PROCESSED_DIR, EMPIRICAL_RESULTS_DIR, SIMULATION_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

def get_baseline_path(n, k):
    return os.path.join(BASELINES_DIR, f"baseline_N{n}_K{k}.json")

# ==========================================
# 3. CORE EXPERIMENT MECHANICS
# ==========================================
ROUNDS = 5                     
NUM_UNIQUE_ITEMS = 160         
MIN_ITEMS_ROUND_N = 4          
MAX_REFILL_LIMIT = 4           
MAX_VISIBLE_ITEMS = 4          

# ==========================================
# 4. SIMULATION SPECIFIC SETTINGS
# ==========================================
N_VALUES = [20, 40, 60, 80, 100]           
K_VALUES = [4, 5]                 
NUM_TRIALS = 100
SIM_TOPICS = ['base_topic', 'side_topic']

PPM_CROSS_EDGE_FRACTIONS = [0.067] 
PRIORITY_QUEUING = True            
LAST_ROUND_PRIORITY = [True]         
USE_EPISTEMIC_NOISE = True 
USE_ALEATORIC_NOISE = True
ALEATORIC_ERROR_RATE = 0.05 

SIDE_TOPIC_ASSIGNMENT_METHOD = 'probabilistic'
SIDE_TOPIC_SPLIT_LEFT_BASE = {'Left': 0.35, 'Center-Left': 0.35, 'Center': 0.23, 'Center-Right': 0.03, 'Right': 0.04}
SIDE_TOPIC_SPLIT_RIGHT_BASE = {'Left': 0.12, 'Center-Left': 0.12, 'Center': 0.12, 'Center-Right': 0.32, 'Right': 0.32}
SIDE_OPINION_VALUES = {'Left': 0.1, 'Center-Left': 0.3, 'Center': 0.5, 'Center-Right': 0.7, 'Right': 0.9}

FILTER_REGIMES = {
    'Hide_Shared': {'hide_shared': True, 'hide_ignored': False}
}

# ==========================================
# 5. BOT SHARING MATRICES PER TOPIC
# ==========================================
SHARE_PROB_MATRIX_BASE = [
    [0.697, 0.600, 0.579, 0.500, 0.176],
    [0.688, 0.938, 0.273, 0.500, 0.286],
    [0.000, 0.000, 0.000, 0.000, 0.000],
    [0.188, 0.417, 0.490, 0.5553, 0.490],
    [0.188, 0.417, 0.490, 0.5553, 0.490]
]

SHARE_PROB_MATRIX_SIDE = [
    [0.647, 0.516, 0.548, 0.333, 0.188],
    [0.750, 0.619, 0.519, 0.571, 0.111],
    [0.490, 0.450, 0.400, 0.470, 0.310],
    [0.429, 0.600, 0.444, 0.500, 0.667],
    [0.148, 0.071, 0.095, 0.462, 0.260]
]

SHARE_PROB_MATRICES = {
    'base_topic': SHARE_PROB_MATRIX_BASE,
    'side_topic': SHARE_PROB_MATRIX_SIDE
}

# ==========================================
# 6. CENTRALIZED CATEGORIZATION MAPPINGS
# ==========================================
USER_CATS = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
ITEM_CATS = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
LEANING_ORDER = ['Left', 'Center Left', 'Center', 'Center Right', 'Right']

def get_bucket(val, is_item=False):
    idx = min(4, max(0, int(val * 5)))
    cats = ITEM_CATS if is_item else USER_CATS
    return cats[idx]

def get_emp_item_bucket(lean_str):
    clean_str = str(lean_str).strip().title()
    if clean_str in ['Left']: return 'Left'
    if clean_str in ['Lean Left', 'Center Left', 'Center-Left']: return 'Center-Left'
    if clean_str in ['Center']: return 'Center'
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

POS_COLORS = {5: '#3498db', 4: '#85c1e9', 3: '#bdc3c7', 2: '#f1948a', 1: '#e74c3c'}
POS_LABELS = {5: 'Left', 4: 'Center Left', 3: 'Center', 2: 'Center Right', 1: 'Right'}
ITEM_MIDPOINTS = {'Left': 0.1, 'Center Left': 0.3, 'Center': 0.5, 'Center Right': 0.7, 'Right': 0.9}
POS_MIDPOINTS = {5: 0.1, 4: 0.3, 3: 0.5, 2: 0.7, 1: 0.9}