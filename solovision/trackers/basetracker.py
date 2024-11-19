import numpy as np
import cv2 as cv
import hashlib
import colorsys
from abc import ABC, abstractmethod
from solovision.utils import logger as LOGGER
from solovision.utils.iou import AssociationFunction


class BaseTracker(ABC):
    def __init__(
        self, 
        nr_classes: int = 80,
        per_class: bool = False,
        asso_func: str = 'iou'
    ):
    
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.last_emb_size = None
        self.asso_func_name = asso_func

        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        
        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []


    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method 
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
    
    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]
            
            if embs is not None:
                # Assert that if embeddings are provided, they have the same number of elements as detections
                assert dets.shape[0] == embs.shape[0], "Detections and embeddings must have the same number of elements when both are provided"
                
                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
                else:
                    class_embs = None
        return class_dets, class_embs
    
    @staticmethod
    def on_first_frame_setup(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """
        def wrapper(self, *args, **kwargs):
            # If setup hasn't been done yet, perform it
            if not self._first_frame_processed:
                img = args[1]
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # Mark that the first frame setup has been done
                self._first_frame_processed = True

            # Call the original method (e.g., update)
            return method(self, *args, **kwargs)
        
        return wrapper
    
    
    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """
        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            
            #handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))
            
            if self.per_class:
                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # Get detections and embeddings for the current class
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                    
                    LOGGER.debug(f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # Activate the specific active tracks for this class id
                    self.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # Reset frame count for every class
                    self.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                    # Save the updated active tracks
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                
                # Increase frame count by 1
                self.frame_count = frame_count + 1

                return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
        return wrapper


    def check_inputs(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

    