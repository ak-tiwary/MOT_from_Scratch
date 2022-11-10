import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from detector import Detector
from track import Track

class Tracker:
    def __init__(self, det, track_expiry_time=50):
        """The main tracker that handles a single step of tracking.
        
        Args:
            det (Detector): A detector object initialized already with desired settings.
            track_expiry_time (int): Number of frames to wait before expiring a track as lost
                                     and starting a new track.
        """
        
        self.alive_tracks = []
        self.tracks_on_hold = []
        self.id_ctr = 0
        self.appearance_model = None #TODO
        
    def step(self, frame):
        """Track objects in new frame."""
        