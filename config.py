import logging
from pathlib import Path

# ==========================================
# 0. REPRODUCIBILITY & OBSERVABILITY
# ==========================================
MASTER_SEED = 42

def setup_logging(log_filename="pipeline_execution.log"):
    """Configures dual-logging to console and a permanent log file."""
    log_path = RESULTS_DIR / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==========================================
# 1. PIPELINE MODE & MASTER SWITCHES
# ==========================================
EMPIRICAL_TOPOLOGY_MODE = 'dual_cross_preasured' #'single_topic', 'dual_independent','dual_cross_preasured' 
EMPIRICAL_BASE_TOPIC = "imm"  
EMPIRICAL_SIDE_TOPIC = "climate"
INVERT_OPINIONS_FOR = ['imm', 'immigration']

# ==========================================
# 2. RUN IDENTIFIERS & NAMING
# ==========================================
CURRENT_EMPIRICAL_TRIAL_ID = "Trial_2026_09_28"
SIMULATION_STUDY_NAME = "pilots45_3x5"

# ==========================================
# 3. DIRECTORY STRUCTURE (pathlib)
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
BASELINES_DIR = DATA_DIR / "baselines"

EMPIRICAL_DATA_DIR = DATA_DIR / "empirical" / CURRENT_EMPIRICAL_TRIAL_ID
EMPIRICAL_RAW_DIR = EMPIRICAL_DATA_DIR / "raw"
EMPIRICAL_PROCESSED_DIR = EMPIRICAL_DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "Results"
EMPIRICAL_RESULTS_DIR = RESULTS_DIR / "Empirical" / CURRENT_EMPIRICAL_TRIAL_ID
SIMULATION_RESULTS_DIR = RESULTS_DIR / "Simulations" / SIMULATION_STUDY_NAME
POWER_RESULTS_DIR = RESULTS_DIR / "Power_Analysis" / SIMULATION_STUDY_NAME

for d in [BASELINES_DIR, EMPIRICAL_RAW_DIR, EMPIRICAL_PROCESSED_DIR, 
          EMPIRICAL_RESULTS_DIR, SIMULATION_RESULTS_DIR, POWER_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def get_baseline_path(n: int, k: int) -> Path:
    return BASELINES_DIR / f"baseline_N{n}_K{k}.json"

def get_sim_sweep_dir(priority_flag: bool, regime_name: str, k: int) -> Path:
    p_str = "Priority_TRUE" if priority_flag else "Priority_FALSE"
    return SIMULATION_RESULTS_DIR / p_str / f"Regime_{regime_name}" / f"K{k}"

def get_power_sweep_dir(priority_flag: bool, regime_name: str, k: int) -> Path:
    p_str = "Priority_TRUE" if priority_flag else "Priority_FALSE"
    return POWER_RESULTS_DIR / p_str / f"Regime_{regime_name}" / f"K{k}"

# ==========================================
# 4. CORE EXPERIMENT MECHANICS
# ==========================================
ROUNDS = 5                     
NUM_UNIQUE_ITEMS = 160         
MIN_ITEMS_ROUND_N = 4          
MAX_REFILL_LIMIT = 4           
MAX_VISIBLE_ITEMS = 4          

# ==========================================
# 5. SIMULATION & POWER ANALYSIS SETTINGS
# ==========================================
N_VALUES = [20, 40, 60, 80, 100]           
K_VALUES = [5]                 
NUM_TRIALS = 10
SIM_TOPICS = ['base_topic', 'side_topic']
DETAILED_SIM_N = [20, 40, 60, 80, 100] 

BASE_TOPIC_EXTREME_FRACTION = 0.9
PPM_CROSS_EDGE_FRACTIONS = [0.067]
PRIORITY_QUEUING = True            
LAST_ROUND_PRIORITY = [True]         

# --- SIMULATION NOISE PARAMETERS ---
SIM_LURKER_FRACTION = 0.15          # Structural: 15% of nodes share 0 items
USE_ALEATORIC_NOISE = True          # Cognitive: Random action injection
ALEATORIC_ERROR_RATE = 0.10         # Cognitive: 5% chance user acts completely randomly

# Monte Carlo Power Analysis Parameters
MC_POOL_SIZE = 500
MC_ITERATIONS = 10000
ANALYSIS_WINDOW_SIZE = 2  # 1 = evaluates ONLY the final round

# --- COMPOSITION NOISE / DEMOGRAPHICS ---
USE_COMPOSITION_NOISE = True        # If True, resamples demographics per trial based on matrix below.
SIDE_TOPIC_ASSIGNMENT_METHOD = 'largest_remainder'

# Master dictionary mapping Base Topic Category -> Probabilities of Side Topic Category.
# Edit these exact numbers if you gather more pilot data or change experimental topics.
EMPIRICAL_COMPOSITION_MATRIX = {
    'Left':   {'Left': 0.70, 'Center-Left': 0.00, 'Center': 0.20, 'Center-Right': 0.00, 'Right': 0.10},
    'Center': {'Left': 0.46, 'Center-Left': 0.00, 'Center': 0.31, 'Center-Right': 0.00, 'Right': 0.23},
    'Right':  {'Left': 0.27, 'Center-Left': 0.00, 'Center': 0.11, 'Center-Right': 0.00, 'Right': 0.62}
}

SIDE_OPINION_VALUES = {'Left': 0.1, 'Center-Left': 0.3, 'Center': 0.5, 'Center-Right': 0.7, 'Right': 0.9}

FILTER_REGIMES = {
    'Hide_Shared': {'hide_shared': True, 'hide_ignored': False}
}

# ==========================================
# 6. EMPIRICAL EXCLUSIONS
# ==========================================
EMPIRICAL_GLOBAL_EXCLUSIONS = []

EMPIRICAL_ROUND_EXCLUSIONS = {
    1: [], 2: [], 3: [], 4: [], 5: [], 6: []
}

# ==========================================
# 7. SHARING PROBABILITY MATRICES
# ==========================================
# NOTE: Absolute 0.00s replaced with 0.05s to prevent perfect firewall bugs.

SHARE_PROB_MATRIX_BASE_SEG = [
    [0.65, 0.60, 0.50, 0.15, 0.30],
    [0.65, 0.60, 0.50, 0.15, 0.30],
    [0.05, 0.05, 0.05, 0.05, 0.05], # Bounded from 0.00
    [0.45, 0.40, 0.55, 0.50, 0.50],
    [0.45, 0.40, 0.55, 0.50, 0.50] 
]

SHARE_PROB_MATRIX_BASE_INT = [
    [0.60, 0.50, 0.45, 0.30, 0.25],  
    [0.60, 0.50, 0.45, 0.30, 0.25],  
    [0.05, 0.05, 0.05, 0.05, 0.05], # Bounded from 0.00 
    [0.25, 0.25, 0.15, 0.50, 0.50],  
    [0.25, 0.25, 0.15, 0.50, 0.50]   
]

SHARE_PROB_MATRIX_SIDE_LEFT_CHAMBER = [
    [0.70, 0.55, 0.35, 0.10, 0.05], # Bounded
    [0.70, 0.55, 0.35, 0.10, 0.05], # Bounded
    [0.90, 0.45, 0.05, 0.95, 0.05],
    [0.85, 0.90, 0.50, 0.50, 0.95],
    [0.85, 0.90, 0.50, 0.50, 0.95]
]

SHARE_PROB_MATRIX_SIDE_RIGHT_CHAMBER = [
    [0.30, 0.45, 0.25, 0.55, 0.40],
    [0.30, 0.45, 0.25, 0.55, 0.40],
    [0.65, 0.50, 0.95, 0.65, 0.90],
    [0.25, 0.10, 0.10, 0.35, 0.50],
    [0.25, 0.10, 0.10, 0.35, 0.50]
]

SHARE_PROB_MATRIX_SIDE_INT = [
    [0.80, 0.65, 0.35, 0.48, 0.15], 
    [0.80, 0.65, 0.35, 0.48, 0.15],  
    [0.75, 0.50, 0.25, 0.35, 0.50],  
    [0.20, 0.25, 0.30, 0.30, 0.10],  
    [0.20, 0.25, 0.30, 0.30, 0.10]
]

SHARE_PROB_MATRICES = {
    'base_topic': {
        'Left': SHARE_PROB_MATRIX_BASE_SEG,
        'Right': SHARE_PROB_MATRIX_BASE_SEG,
        'integrated': SHARE_PROB_MATRIX_BASE_INT
    },
    'side_topic': {
        'Left': SHARE_PROB_MATRIX_SIDE_LEFT_CHAMBER,
        'Right': SHARE_PROB_MATRIX_SIDE_RIGHT_CHAMBER,
        'integrated': SHARE_PROB_MATRIX_SIDE_INT
    }
}

# ==========================================
# 8. MAPPINGS & PRESENTATION PALETTES
# ==========================================
USER_CATS = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
ITEM_CATS = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
LEANING_ORDER = ['Left', 'Center Left', 'Center', 'Center Right', 'Right']

POS_COLORS = {5: '#3498db', 4: '#85c1e9', 3: '#bdc3c7', 2: '#f1948a', 1: '#e74c3c'}
POS_LABELS = {5: 'Left', 4: 'Center Left', 3: 'Center', 2: 'Center Right', 1: 'Right'}
ITEM_MIDPOINTS = {'Left': 0.1, 'Center Left': 0.3, 'Center': 0.5, 'Center Right': 0.7, 'Right': 0.9}
POS_MIDPOINTS = {5: 0.1, 4: 0.3, 3: 0.5, 2: 0.7, 1: 0.9}

# ==========================================
# 9. OPERATIONAL THRESHOLDS & ALGORITHM PARAMETERS
# ==========================================
REWIRE_SWAPS_MULTIPLIER = 10
REWIRE_TRIES_MULTIPLIER = 100

CONGENIAL_THRESHOLD_LOWER = 0.4
CONGENIAL_THRESHOLD_UPPER = 0.6

# Data Contract Schema (Documentation)
EVENT_LOG_SCHEMA = {
    'trial_id': 'string', 'topic': 'string', 'user_id': 'string', 
    'treatment': 'string', 'chamber': 'string', 'op_raw': 'float', 
    'user_cat': 'string', 'contrary_neighbors': 'int', 'round': 'int', 
    'item_id': 'string', 'item_cat': 'string', 'action': 'string', 
    'source': 'string', 'is_backlog': 'bool'
}