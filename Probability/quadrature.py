import slicer
import numpy
import vtk
from scipy.stats import norm
from itertools import product
from vtk.util.numpy_support import vtk_to_numpy

#Radius
ICESEED_R = 7.5
ICESEED_A = 7.5
ICESEED_S = 15.0

class quad:


    def createPerturbedCenters(self):
        num_nodes = 5
        perturbed_x, weights = self.gaussian_quadrature_nodes(num_nodes, center[0], SDinplane)
        perturbed_y, weights = self.gaussian_quadrature_nodes(num_nodes, center[1], SDinplane)
        perturbed_z, weights = self.gaussian_quadrature_nodes(num_nodes, center[2], SDdepth)

    def runQuadrature(self,center,inputVolume,SDinplane,SDdepth,nOfPoints):
        num_nodes = 5
        coordinates = []
        centers = []
        for i in range(nOfPoints):
            centers.append([center[0+i*3],center[1+i*3],center[2+i*3]])
            perturbed_x, weights = self.gaussian_quadrature_nodes(num_nodes, center[0+i*3], SDinplane)
            perturbed_y, weights = self.gaussian_quadrature_nodes(num_nodes, center[1+i*3], SDinplane)
            perturbed_z, weights = self.gaussian_quadrature_nodes(num_nodes, center[2+i*3], SDdepth)
            coordinate = self.create_coordinates_new(perturbed_x, perturbed_y, perturbed_z)
            coordinates.append(coordinate)

        #Parei aqui.
        #perturbed_x, weights = self.gaussian_quadrature_nodes(num_nodes, center[0], SDinplane)
        #perturbed_y, weights = self.gaussian_quadrature_nodes(num_nodes, center[1], SDinplane)
        #perturbed_z, weights = self.gaussian_quadrature_nodes(num_nodes, center[2], SDdepth)

        #coordinates = self.create_coordinates_new(perturbed_x, perturbed_y, perturbed_z)
        return self.perturbed_tumor_percentage(inputVolume, coordinates, [ICESEED_R, ICESEED_A, ICESEED_S],weights)

    def gaussian_quadrature_nodes_weights(self,n):
        # Generate nodes and weights for Legendre-Gauss quadrature
        nodes, weights = numpy.polynomial.legendre.leggauss(n)

        #nodes, weights = numpy.polynomial.hermite.hermgauss(n)
        # Map nodes to the interval [-1, 1]
        #nodes = nodes * 2 - 1
        return nodes, weights

    def create_coordinates_new(self,x_list, y_list, z_list):
        # Combine all elements from the lists without repetition
        coordinates = [(x_list[0],y_list[0],z_list[0]),(x_list[1],y_list[1],z_list[1]),(x_list[2],y_list[2],z_list[2]),(x_list[3],y_list[3],z_list[3]),(x_list[4],y_list[4],z_list[4])]
        return coordinates


    def create_coordinates(self,x_list, y_list, z_list):
        # Combine all elements from the lists without repetition
        coordinates = list(product(x_list, y_list, z_list))
        return coordinates

    def perturbed_tumor_percentage(self,nrrd, coordinates, radii, weights):
        tumor_percentages = []

        for i in range(len(coordinates)):
            center = coordinates[i]
            percentage = self.iceball_coverage_1(nrrd, center)
            tumor_percentages.append(percentage * weights[i])
        return numpy.sum(tumor_percentages)/numpy.sum(weights)

    def gaussian_quadrature_nodes(self,num_nodes,center,sd):

        nodes, weights = self.gaussian_quadrature_nodes_weights(num_nodes)
        # Perturb each node value by the specified standard deviation
        perturbed_nodes = center + nodes * sd
        # Return the perturbed node values
        return perturbed_nodes, weights

    def getIceballImageData(self,center,Spacing,size_image):
        sphere1 = vtk.vtkImageEllipsoidSource()
        sphere1.SetOutputScalarTypeToShort()
        sphere1.SetCenter([center[0],center[1],center[2]])
        sphere1.SetRadius(ICESEED_R / Spacing[0], ICESEED_A / Spacing[1], ICESEED_S / Spacing[2])
        sphere1.SetWholeExtent(0, size_image[0] - 1, 0, size_image[1] - 1, 0, size_image[2] - 1)
        sphere1.Update()
        return sphere1.GetOutput()

    def andLogic(self,image,sp):
        logic = vtk.vtkImageLogic()
        logic.SetInput1Data(image)
        logic.SetInput2Data(sp)
        logic.SetOperationToAnd()
        logic.SetOutputTrueValue(1)
        logic.Update()
        return logic.GetOutput()

    def addIceballToScene(self,image,IjkToRasMatrix,data):
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

    def iceball_coverage_1(self,reader, center):

        RASToIJKMatrix = vtk.vtkMatrix4x4()
        reader.GetRASToIJKMatrix(RASToIJKMatrix)
        position = [center[0],center[1],center[2],1]
        center_probe = RASToIJKMatrix.MultiplyPoint(position)
        image = reader.GetImageData()
        size_image = image.GetDimensions()
        Spacing = reader.GetSpacing()

        sp = self.getIceballImageData(center_probe, Spacing, size_image)
        fusedImage = self.andLogic(image,sp)

        # Extract scalar data and calculate iceball coverage
        image_data = vtk_to_numpy(image.GetPointData().GetScalars())
        logic_data = vtk_to_numpy(fusedImage.GetPointData().GetScalars())

        image_coverage = numpy.sum(image_data)  # Total coverage of the original image
        iceball_coverage = numpy.sum(logic_data)  # Coverage of the iceball within the image

        IjkToRasMatrix = vtk.vtkMatrix4x4()
        reader.GetIJKToRASMatrix(IjkToRasMatrix)
        self.addIceballToScene(reader, IjkToRasMatrix, sp)

        return iceball_coverage / image_coverage
