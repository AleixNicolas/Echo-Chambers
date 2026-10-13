import pytest
from src.simulation.core_engine import Participant
import config
import random

def test_draw_feed_fallback_cascade():
    """
    Validates that draw_feed correctly exhausts queues in the exact strict order:
    1. Last round (never seen)
    2. Previous rounds (never seen)
    3. Last round (seen)
    4. Previous rounds (seen)
    """
    # 1. Rig the global configurations for testing
    config.MAX_VISIBLE_ITEMS = 4
    config.SIM_TOPICS = ['test_topic']
    
    # Fake global item pool
    fake_pool = {'test_topic': {
        'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4, 
        'E': 0.5, 'F': 0.6, 'G': 0.7, 'H': 0.8
    }}
    
    # 2. Instantiate a Participant
    p = Participant(node_id=1, opinions={'test_topic': 0.5}, run_matrices={'test_topic': [[0.5]*5]*5}, starting_items=[])
    
    # 3. Rig the Participant's exact state
    # Items they have seen in the past
    p.history['test_topic'] = {'A', 'B', 'C'}
    
    # Items that arrived in their queue *this round* (some seen, some novel)
    p.incoming_queue['test_topic'] = {
        'A': 1,  # Seen last round
        'D': 1   # Novel last round
    }
    
    # Items resting in the backlog from *previous rounds* (some seen, some novel)
    p.backlog['test_topic'] = {
        'B': 1,  # Seen previously
        'E': 1   # Novel previously
    }
    
    # 4. Execute the draw
    random.seed(42) # Lock randomness
    feed, source_map = p.draw_feed('test_topic', round_num=2, item_pool=fake_pool, hide_shared=False, hide_ignored=False)
    
    # 5. Assertions (The Mathematical Proof)
    # The output feed MUST be exactly 4 items, in exactly this priority order:
    # 1. D (Last round, never seen)
    # 2. E (Prev round, never seen)
    # 3. A (Last round, seen)
    # 4. B (Prev round, seen)
    
    assert len(feed) == 4, "Feed did not draw exactly 4 items."
    
    # Verify the cascade priority strictly enforced the order
    assert 'D' in feed, "Failed to prioritize: Last round (never seen)"
    assert 'E' in feed, "Failed to prioritize: Previous rounds (never seen)"
    assert 'A' in feed, "Failed to prioritize: Last round (seen)"
    assert 'B' in feed, "Failed to prioritize: Previous rounds (seen)"
    
    # Verify the source tracking accurately mapped the is_backlog flag
    assert source_map['D'] == ('neighbor', False)
    assert source_map['E'] == ('neighbor', True)
    assert source_map['A'] == ('neighbor', False)
    assert source_map['B'] == ('neighbor', True)