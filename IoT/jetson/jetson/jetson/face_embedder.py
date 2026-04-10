import cv2
import numpy as np 
import onnxruntime as ort


class FaceEmbedder:
    def __init__(self, model_path="modeling/mobilefacenet.onnx"):
        """Class which will load a model of MobileFaceNet and will transform image into embedded vector"""
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # We assign expected size of the input for the model, because MobileFaceNet expects 112 x 112
        self.input_shape = self.session.get_inputs()[0].shape[2:]


    def get_embedding(self, face_image):
        """Function which transforms image (BGR) into vector embedding"""
        # Scaling to the expected size for the model
        resized = cv2.resize(face_image, (self.input_shape[1], self.input_shape[0]))

        # Preprocessing - conversion BGR -> RGB and normalization to the boundariues [-1, 1]
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_normlized = (np.float32(rgb_image) - 127.5 / 128)

        # Change the shape (Height, Width, Channel) -> (1, Channel, Height, Width) for model
        img_transposed = np.transpose(img_normlized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        # Creating embedding, running inference
        # First arg its a list of oututs of the model - None means "give me all outputs"
        # Second arg is dict of input
        embedding = self.session.run(None, {self.input_name: img_batch})[0]
        embedding = embedding.flatten()

        # Return normalized vector
        return embedding / np.linalg.norm(embedding)