import logging
from enum import Enum, auto

import cuvis_ai
from cuvis import Measurement

import numpy as np

class Classification:
    class Model(Enum):
        EFFICIENTAD = auto()
        CPR = auto()
        NONE = auto()
        
    def __init__(self, model: Model):
        self.logger = logging.getLogger("cla")
        # read model
        match model:
            case self.Model.EFFICIENTAD:
                self.logger.info("Choosing Model \"EFFICIENTAD\".")
                self.graph = cuvis_ai.pipeline.Graph.load_from_file('models/efficientad.zip')
            case self.Model.CPR:
                self.logger.info("Choosing Model \"CPR\".")
                self.graph = cuvis_ai.pipeline.Graph.load_from_file('models/cpr.zip')
            case self.Model.NONE:
                self.logger.warning("Choosing Model \"NONE\".")
                self.graph = None
            case _:
                raise ValueError("unknown model")
        self.logger.debug("Classification object created.")
    
    def __del__(self):
        self.logger.debug("Classification object destroyed.")

    async def classify(self, measurement: Measurement) -> np.ndarray:
        # run model
        if self.graph is not None:
            self.logger.info("Evaluating Cube.")
            input_data = measurement.cube.array
            input_data = np.expand_dims(input_data, 0)
            output = np.squeeze(self.graph.forward(input_data))
            return output
        else:
            self.logger.warning("Model is NONE. Cannot process. Returning BS.")
            return None