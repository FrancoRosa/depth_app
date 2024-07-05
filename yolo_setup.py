# Description: This script demonstrates how to perform inference on DepthAI using OpenVINO model.

import cv2

import depthai as dai
import numpy as np
import time
import blobconverter
# from utility import *

NN_SIZE = (640,640)
text_color = (0, 0, 255)
bbox_color = (0, 255, 0)
# Define the path to the OpenVINO model
nnBlobPath = "weights_blob/best_openvino_2022.1_6shave.blob"
# Define the labels for the model
LABEL_MAP_YOLO = ['crate', 'junction box', 'module', 'panel', 'person', 'pile', 'pipe', 'pipes', 'tube']

def setup_pipeline_yolo():
    pipeline = dai.Pipeline()

    # Create output streams
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    # Setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(True)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())


    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(NN_SIZE)
    camRgb.setInterleaved(False)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setIspScale(1,3)
    # camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # camRgb.setPreviewKeepAspectRatio(False)

    # Define the output streams
    xoutRgb = pipeline.create(dai.node.XLinkOut)



    xoutRgb.setStreamName("rgb")



    camRgb.video.link(xoutRgb.input)




    yolo_spatial_det_nn = pipeline.createYoloSpatialDetectionNetwork()
    # Define the model properties
    yolo_spatial_det_nn.setConfidenceThreshold(0.5)
    yolo_spatial_det_nn.setBlobPath(nnBlobPath)
    yolo_spatial_det_nn.setNumClasses(9)  # Adjust based on your model
    yolo_spatial_det_nn.setCoordinateSize(4)
    yolo_spatial_det_nn.setAnchors([])  # Adjust based on your model
    yolo_spatial_det_nn.setAnchorMasks({})  # Adjust based on your model
    yolo_spatial_det_nn.setIouThreshold(0.5)
    yolo_spatial_det_nn.setDepthLowerThreshold(100)
    yolo_spatial_det_nn.setDepthUpperThreshold(5000)

    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("detections")
    yolo_spatial_det_nn.out.link(xoutNN.input)
    camRgb.preview.link(yolo_spatial_det_nn.input)
    


    passthroughOut = pipeline.create(dai.node.XLinkOut)

    passthroughOut.setStreamName("pass")


    stereo.depth.link(yolo_spatial_det_nn.inputDepth)
    yolo_spatial_det_nn.passthroughDepth.link(xoutDepth.input)
    yolo_spatial_det_nn.passthrough.link(passthroughOut.input)

    return pipeline