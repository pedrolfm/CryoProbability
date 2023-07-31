import logging
import os
from vtk.util.numpy_support import vtk_to_numpy
import vtk
import numpy
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
#from quadrature import quad
from scipy.stats import norm
from itertools import product
import time


#
# Probability
#

class Probability(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Probability"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Pedro Moreira and Alvaro Cuervo"]
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Probability">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Probability1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Probability',
        sampleName='Probability1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'Probability1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='Probability1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='Probability1'
    )

    # Probability2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Probability',
        sampleName='Probability2',
        thumbnailFileName=os.path.join(iconsPath, 'Probability2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='Probability2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='Probability2'
    )


#
# ProbabilityWidget
#

class ProbabilityWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Probability.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ProbabilityLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputProbeLocation.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputProbeLocation.setMRMLScene(slicer.mrmlScene)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True


        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        t1 = time.time()
        percentage,prob = self.logic.runProbability(self.ui.inputProbeLocation.currentNode(),self.ui.inputSelector.currentNode(),self.ui.errorInPlane.value,self.ui.errorDepth.value)
        t2 = time.time()
        self.ui.resultsLabel.setText("Percentage coverage: "+str(percentage)+"\n"+"Probability: "+str(prob*100)+"% \n Time: "+str(t2-t1)+" seconds \n")

#
# ProbabilityLogic
#

class ProbabilityLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.percentage = quad()

    def iceball_coverage_1(self,reader, center, radii):
        # Read the NRRD file using vtkNrrdReader

        RASToIJKMatrix = vtk.vtkMatrix4x4()
        reader.GetRASToIJKMatrix(RASToIJKMatrix)
        position = [center[0],center[1],center[2],1]
        center_probe = RASToIJKMatrix.MultiplyPoint(position)
        image = reader.GetImageData()

        # Create the ellipsoid source
        sphere1 = vtk.vtkImageEllipsoidSource()
        sphere1.SetOutputScalarTypeToShort()
        sphere1.SetCenter([center_probe[0],center_probe[1],center_probe[2]])
        Spacing = reader.GetSpacing()
        print(Spacing)
        sphere1.SetRadius(radii[0] / Spacing[0], radii[1] / Spacing[1], radii[2] / Spacing[2])

        # Create the dimensions of the file
        size_image = image.GetDimensions()
        print(size_image)
        sphere1.SetWholeExtent(0, size_image[0] - 1, 0, size_image[1] - 1, 0, size_image[2] - 1)
        sphere1.Update()

        # Align the iceball and tumor images
        sp = sphere1.GetOutput()
        sp.SetOrigin(image.GetOrigin())

        # Logic AND operation between iceball and tumor
        logic = vtk.vtkImageLogic()
        logic.SetInput1Data(image)
        logic.SetInput2Data(sp)
        logic.SetOperationToAnd()
        logic.SetOutputTrueValue(1)
        logic.Update()

        # Extract scalar data and calculate iceball coverage
        image_data = vtk_to_numpy(image.GetPointData().GetScalars())
        logic_data = vtk_to_numpy(logic.GetOutput().GetPointData().GetScalars())

        image_coverage = numpy.sum(image_data)  # Total coverage of the original image
        iceball_coverage = numpy.sum(logic_data)  # Coverage of the iceball within the image

        IjkToRasMatrix = vtk.vtkMatrix4x4()
        reader.GetIJKToRASMatrix(IjkToRasMatrix)
        outputLabelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        outputLabelmapVolumeNode.SetOrigin(reader.GetOrigin())
        outputLabelmapVolumeNode.SetSpacing(reader.GetSpacing())
        outputLabelmapVolumeNode.SetIJKToRASMatrix(IjkToRasMatrix)
        outputLabelmapVolumeNode.SetAndObserveImageData(logic.GetOutput())
        outputLabelmapVolumeNode.CreateDefaultDisplayNodes()
        outputLabelmapVolumeNode.SetName("imageteste")

        return iceball_coverage / image_coverage

    def perturbed_tumor_percentage(self,nrrd, coordinates, sd_x, sd_y, sd_z, radii):
        for i in range(len(coordinates)):
            center = coordinates #for just one probe
            percentage = self.iceball_coverage_1(nrrd, center, radii)
        return percentage

    def runProbability(self,location,tumor,sd1,sd2):
        #location = slicer.util.getNode('F_1')
        ras_target = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]
        probeLocation = []
        nOfFiducials = location.GetNumberOfFiducials()
        for n in range(nOfFiducials):
            location.GetNthFiducialPosition(n, ras_target[n])
        #prob = self.percentage.runQuadrature(ras_target,tumor,sd1,sd2,nOfFiducials)
        return self.percentage.runQuadrature(ras_target,tumor,sd1,sd2,nOfFiducials)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# Temporary here to make reload easier.
#

ICESEED_R = 12.0
ICESEED_A = 12.0
ICESEED_S = 15.0


class quad:

    def createPerturbedCenters(self):
        num_nodes = 5
        perturbed_x, weights = self.gaussian_quadrature_nodes(num_nodes, center[0], SDinplane)
        perturbed_y, weights = self.gaussian_quadrature_nodes(num_nodes, center[1], SDinplane)
        perturbed_z, weights = self.gaussian_quadrature_nodes(num_nodes, center[2], SDdepth)

    def runQuadrature(self, center, inputVolume, SDinplane, SDdepth, nOfPoints):
        num_nodes = 5
        coordinates = []
        coord = numpy.array([[0.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0],[0.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0],[0.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0]])
        for i in range(nOfPoints):
            #centers.append([center[0 + i * 3], center[1 + i * 3], center[2 + i * 3]])
            perturbed_x, weights = self.gaussian_quadrature_nodes(num_nodes, center[i][0], SDinplane)
            perturbed_y, weights = self.gaussian_quadrature_nodes(num_nodes, center[i][1], SDinplane)
            perturbed_z, weights = self.gaussian_quadrature_nodes(num_nodes, center[i][2], SDdepth)
            coordinate = self.create_coordinates_new(perturbed_x, perturbed_y, perturbed_z)
            for j in range(num_nodes):
                coord[i][0+j*3] = coordinate[j][0]
                coord[i][1+j*3] = coordinate[j][1]
                coord[i][2+j*3] = coordinate[j][2]
        # Parei aqui.
        # perturbed_x, weights = self.gaussian_quadrature_nodes(num_nodes, center[0], SDinplane)
        # perturbed_y, weights = self.gaussian_quadrature_nodes(num_nodes, center[1], SDinplane)
        # perturbed_z, weights = self.gaussian_quadrature_nodes(num_nodes, center[2], SDdepth)
        # coordinates = self.create_coordinates_new(perturbed_x, perturbed_y, perturbed_z)
        return self.perturbed_tumor_percentage(inputVolume, coord, [ICESEED_R, ICESEED_A, ICESEED_S], weights,nOfPoints)

    def gaussian_quadrature_nodes_weights(self, n):
        # Generate nodes and weights for Legendre-Gauss quadrature
        nodes, weights = numpy.polynomial.legendre.leggauss(n)

        # nodes, weights = numpy.polynomial.hermite.hermgauss(n)
        # Map nodes to the interval [-1, 1]
        nodes = nodes * 2 - 1
        return nodes, weights

    def create_coordinates_new(self, x_list, y_list, z_list):
        # Combine all elements from the lists without repetition
        coordinates = [(x_list[0], y_list[0], z_list[0]), (x_list[1], y_list[1], z_list[1]),
                       (x_list[2], y_list[2], z_list[2]), (x_list[3], y_list[3], z_list[3]),
                       (x_list[4], y_list[4], z_list[4])]
        return coordinates

    def create_coordinates(self, x_list, y_list, z_list):
        # Combine all elements from the lists without repetition
        coordinates = list(product(x_list, y_list, z_list))
        return coordinates

    def perturbed_tumor_percentage(self, nrrd, coordinates, radii, weights,nOfProbes):
        tumor_percentages = []
        prob = []
        for j in range(5): #number of nodes - change
            center = [[coordinates[0][0+j*3],coordinates[0][1+j*3],coordinates[0][2+j*3]],[coordinates[1][0+j*3],coordinates[1][1+j*3],coordinates[1][2+j*3]],[coordinates[2][0+j*3],coordinates[2][1+j*3],coordinates[2][2+j*3]]]
            percentage = self.iceball_coverage_1(nrrd, center,nOfProbes)
            tumor_percentages.append(percentage * weights[j])
            if percentage >= 0.99:
                prob.append(weights[j])
            else:
                prob.append(0.0)
        print(tumor_percentages)
        return numpy.sum(tumor_percentages) / numpy.sum(weights), numpy.sum(prob)/ numpy.sum(weights)

    def gaussian_quadrature_nodes(self, num_nodes, center, sd):
        nodes, weights = self.gaussian_quadrature_nodes_weights(num_nodes)
        # Perturb each node value by the specified standard deviation
        perturbed_nodes = center + nodes * sd
        # Return the perturbed node values
        return perturbed_nodes, weights

    def getIceballImageData(self, center, Spacing, size_image):
        sphere1 = vtk.vtkImageEllipsoidSource()
        sphere1.SetOutputScalarTypeToShort()
        sphere1.SetCenter([center[0], center[1], center[2]-3])
        sphere1.SetRadius(ICESEED_R / Spacing[0], ICESEED_A / Spacing[1], ICESEED_S / Spacing[2])
        sphere1.SetWholeExtent(0, size_image[0] - 1, 0, size_image[1] - 1, 0, size_image[2] - 1)
        sphere1.Update()
        return sphere1.GetOutput()

    def andLogic(self, image, sp):
        logic = vtk.vtkImageLogic()
        logic.SetInput1Data(image)
        logic.SetInput2Data(sp)
        logic.SetOperationToAnd()
        logic.SetOutputTrueValue(1)
        logic.Update()
        return logic.GetOutput()


    def orLogic(self, image, sp):
        logic = vtk.vtkImageLogic()
        logic.SetInput1Data(image)
        logic.SetInput2Data(sp)
        logic.SetOperationToOr()
        logic.SetOutputTrueValue(1)
        logic.Update()
        return logic.GetOutput()

    def addIceballToScene(self, image, IjkToRasMatrix, data):
        try:
            outputLabelmapVolumeNode = slicer.util.getNode('iceball')
        except:
            outputLabelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            outputLabelmapVolumeNode.SetName("iceball")

        outputLabelmapVolumeNode.SetOrigin(image.GetOrigin())
        outputLabelmapVolumeNode.SetSpacing(image.GetSpacing())
        outputLabelmapVolumeNode.SetIJKToRASMatrix(IjkToRasMatrix)
        outputLabelmapVolumeNode.SetAndObserveImageData(data)
        outputLabelmapVolumeNode.CreateDefaultDisplayNodes()
        slicer.util.setSliceViewerLayers(outputLabelmapVolumeNode, fit=True)

    def iceball_coverage_1(self, reader, center, nOfProbes):

        RASToIJKMatrix = vtk.vtkMatrix4x4()
        reader.GetRASToIJKMatrix(RASToIJKMatrix)
        IjkToRasMatrix = vtk.vtkMatrix4x4()
        reader.GetIJKToRASMatrix(IjkToRasMatrix)
        
        image = reader.GetImageData()
        image.AllocateScalars(4, 1)
        size_image = image.GetDimensions()
        Spacing = reader.GetSpacing()

        position = [center[0][0], center[0][1], center[0][2], 1]
        center_probe = RASToIJKMatrix.MultiplyPoint(position)
        sp = self.getIceballImageData(center_probe, Spacing, size_image)

  
        for i in range(nOfProbes):
            position = [center[i][0], center[i][1], center[i][2], 1]
            center_probe = RASToIJKMatrix.MultiplyPoint(position)
            sp2 = self.getIceballImageData(center_probe, Spacing, size_image)
            sp = self.orLogic(sp, sp2)

        self.addIceballToScene(reader, IjkToRasMatrix, sp)

        fusedImage = self.andLogic(image, sp)

        # Extract scalar data and calculate iceball coverage
        image_data = vtk_to_numpy(image.GetPointData().GetScalars())
        logic_data = vtk_to_numpy(fusedImage.GetPointData().GetScalars())

        image_coverage = numpy.sum(image_data)  # Total coverage of the original image
        iceball_coverage = numpy.sum(logic_data)  # Coverage of the iceball within the image

        return iceball_coverage / image_coverage


#################################################
#################################################





class ProbabilityTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_Probability1()

    def test_Probability1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('Probability1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = ProbabilityLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
