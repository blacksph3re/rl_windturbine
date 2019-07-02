# -------------------------------------------------
# Project created by David Marten using QtCreator
# -------------------------------------------------

# switch here to build either for 32bit or 64bit
#CONFIG += build32bit
CONFIG += build64bit

# set USEGUI=false to compile the console application
# unfortunately its not possible to get cmd output whithout console option
# if a GUI application is compiled, this will open a CMS window on start which is not desirable, the only convenient
# solution is to compile GUI and command line versions seperately
# ATTENTION: make sure to rebuild Main.cpp when changing this value as qmake doesn't get the dependency
DEFINES += USEGUI=false
!USEGUI: CONFIG += console

# specify which Qt modules are needed.
QT += core gui widgets opengl xml testlib

# set the name of the executable
TARGET = QBlade

# The template to use for the project. This determines that the output will be a dll.
TEMPLATE = lib
DEFINES += QBLADE_LIBRARY=true

# The template to use for the project. This determines that the output will be an application.
#TEMPLATE = app

# include the resources file into the binary
RESOURCES += qblade.qrc

# set the icon of the executable application file
win32:RC_FILE = win/qblade.rc

# from gcc 4.8.1 on the c++11 implementation is feature complete
QMAKE_CXXFLAGS += -std=gnu++11  # usually it would be fine to give c++11 but the OpenCL part apparently needs gnu extensions
QMAKE_CXXFLAGS += -fpermissive

# activate compiler support for openMP
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

# add the proper include path for libraries
     win32:CONFIG(build32bit): LIBS += -L$$PWD/libs_windows_32bit
else:win32:CONFIG(build64bit): LIBS += -L$$PWD/libs_windows_64bit
else:unix: CONFIG(build32bit): LIBS += -L$$PWD/libs_unix_32bit
else:unix: CONFIG(build64bit): LIBS += -L$$PWD/libs_unix_64bit


# includes QGLViewer
INCLUDEPATH += include_qglviewer
     win32:CONFIG(release, debug|release): LIBS += -lQGLViewer2
else:win32:CONFIG(debug,   debug|release): LIBS += -lQGLViewerd2
else:unix: LIBS += -lQGLViewer

# include openGL & glu
     win32: LIBS += -lopengl32 -lglu32
else:unix:  LIBS += -lGLU

# include openCL
INCLUDEPATH += include_opencl
     win32: LIBS += -lOpenCL
else:unix:  LIBS += -lOpenCL

# include mkl
INCLUDEPATH += include_mkl/
win32: LIBS += -lmkl_rt
unix:  LIBS += -lmkl_rt -L/opt/intel/mkl/lib/intel64
win32: LIBS += -lmkl_intel_thread
win32: LIBS += -lmkl_core
win32: LIBS += -lmkl_avx2

# include Project Chrono
INCLUDEPATH += include_chrono
INCLUDEPATH += include_chrono/src
INCLUDEPATH += include_chrono/src/chrono/
INCLUDEPATH += include_chrono/src/chrono/collision/
INCLUDEPATH += include_chrono/src/chrono/collision/bullet/

win32: LIBS += -llibChronoEngine
else:unix: LIBS += -lChronoEngine 
win32: LIBS += -llibChronoEngine_mkl
else:unix: LIBS += -lChronoEngine_mkl


# include clBlast
INCLUDEPATH += include_CLBlast
win23: LIBS += -llibclblast
else:unix: LIBS += -lclblast

SOURCES += src/MainFrame.cpp \
    src/Main.cpp \
    src/Globals.cpp \
    src/TwoDWidget.cpp \
    src/GLWidget.cpp \
    src/Misc/GLLightDlg.cpp \
    src/XDirect/BatchDlg.cpp \
    src/XDirect/BatchThreadDlg.cpp \
    src/XDirect/XFoilTask.cpp \
    src/XDirect/CAddDlg.cpp \
    src/XDirect/FoilCoordDlg.cpp \
    src/XDirect/FoilGeomDlg.cpp \
    src/XDirect/FoilPolarDlg.cpp \
    src/XDirect/FlapDlg.cpp \
    src/XDirect/InterpolateFoilsDlg.cpp \
    src/XDirect/LEDlg.cpp \
    src/XDirect/ManageFoilsDlg.cpp \
    src/XDirect/NacaFoilDlg.cpp \
    src/XDirect/ReListDlg.cpp \
    src/XDirect/TEGapDlg.cpp \
    src/XDirect/TwoDPanelDlg.cpp \
    src/XDirect/XDirectStyleDlg.cpp \
    src/XDirect/XFoil.cpp \
    src/XDirect/XFoilAnalysisDlg.cpp \
    src/XDirect/XFoilAdvancedDlg.cpp \
    src/XDirect/XDirect.cpp \
    src/Objects/CVector.cpp \
    src/Objects/Foil.cpp \
    src/Objects/OpPoint.cpp \
    src/VortexObjects/Panel.cpp \
    src/Objects/Polar.cpp \
    src/Objects/Quaternion.cpp \
    src/Objects/Sf.cpp \
    src/Objects/Spline.cpp \
    src/Misc/EditPlrDlg.cpp \
    src/Misc/PolarFilterDlg.cpp \
    src/Misc/TranslatorDlg.cpp \
    src/Misc/UnitsDlg.cpp \
    src/Misc/LinePickerDlg.cpp \
    src/Misc/LineDelegate.cpp \
    src/Misc/LineCbBox.cpp \
    src/Misc/LineButton.cpp \
    src/Misc/FloatEditDelegate.cpp \
    src/Misc/DisplaySettingsDlg.cpp \
    src/Misc/ColorButton.cpp \
    src/Misc/AboutQ5.cpp \
    src/Misc/ObjectPropsDlg.cpp \
    src/Graph/QGraph.cpp \
    src/Graph/GraphWidget.cpp \
    src/Graph/GraphDlg.cpp \
    src/Graph/Graph.cpp \
    src/Graph/Curve.cpp \
    src/XInverse/FoilSelectionDlg.cpp \
    src/XInverse/PertDlg.cpp \
    src/XInverse/XInverse.cpp \
    src/XInverse/InverseOptionsDlg.cpp \
    src/Design/FoilTableDelegate.cpp \
    src/Design/AFoilGridDlg.cpp \
    src/Design/LECircleDlg.cpp \
    src/Design/AFoil.cpp \
    src/Design/SplineCtrlsDlg.cpp \
    src/Design/AFoilTableDlg.cpp \
    src/XBEM/TData.cpp \
    src/XBEM/TBEMData.cpp \
    src/XBEM/SimuWidget.cpp \
    src/XBEM/OptimizeDlg.cpp \
    src/XBEM/Edit360PolarDlg.cpp \
    src/XBEM/CreateBEMDlg.cpp \
    src/XBEM/BladeScaleDlg.cpp \
    src/XBEM/BEMData.cpp \
    src/XBEM/BEM.cpp \
    src/XBEM/BData.cpp \
    src/XBEM/AboutBEM.cpp \
    src/XBEM/Blade.cpp \
    src/Objects/Surface.cpp \
    src/XBEM/BladeDelegate.cpp \
    src/XBEM/BladeAxisDelegate.cpp \
    src/XBEM/CBEMData.cpp \
    src/XBEM/PrescribedValuesDlg.cpp \
    src/XBEM/CircularFoilDlg.cpp \
    src/XBEM/BEMSimDock.cpp \
    src/XBEM/BEMDock.cpp \
    src/XDMS/DMS.cpp \
    src/XDMS/SimuWidgetDMS.cpp \
    src/XDMS/BladeDelegate2.cpp \
    src/XDMS/OptimizeDlgVAWT.cpp \
    src/XDMS/BladeScaleDlgVAWT.cpp \
    src/XDMS/CreateDMSDlg.cpp \
    src/XDMS/DMSData.cpp \
    src/XDMS/DData.cpp \
    src/XDMS/TDMSData.cpp \
    src/XDMS/DMSSimDock.cpp \
    src/XDMS/DMSDock.cpp \
    src/XDMS/CDMSData.cpp \
    src/XUnsteadyBEM/WindField.cpp \
    src/XWidgets.cpp \
    src/XUnsteadyBEM/WindFieldModule.cpp \
    src/Module.cpp \
    src/XUnsteadyBEM/WindFieldToolBar.cpp \
    src/ScrolledDock.cpp \
    src/Store.cpp \
    src/XUnsteadyBEM/WindFieldMenu.cpp \
    src/QFem/taperedelem.cpp \
    src/QFem/structintegrator.cpp \
    src/QFem/structelem.cpp \
    src/QFem/node.cpp \
    src/QFem/eqnmotion.cpp \
    src/QFem/clipper.cpp \
    src/QFem/unitvector.cpp \
    src/QFem/mode.cpp \
    src/QFem/deformationvector.cpp \
    src/QFem/QFEMDock.cpp \
    src/QFem/QFEMToolBar.cpp \
    src/QFem/QFEMModule.cpp \
    src/QFem/QFEMMenu.cpp \
    src/QFem/StructDelegate.cpp \
    src/QFem/BladeStructure.cpp \
    src/StorableObject.cpp \
    src/XUnsteadyBEM/FASTModule.cpp \
    src/XUnsteadyBEM/FASTDock.cpp \
    src/XUnsteadyBEM/FASTSimulation.cpp \
    src/XUnsteadyBEM/FASTSimulationCreatorDialog.cpp \
    src/XUnsteadyBEM/FASTSimulationToolBar.cpp \
    src/XUnsteadyBEM/FASTSimulationProgressDialog.cpp \
    src/XBEM/BEMToolbar.cpp \
    src/XDMS/DMSToolbar.cpp \
    src/XBEM/360Polar.cpp \
    src/Misc/NumberEdit.cpp \
    src/Serializer.cpp \
    src/StoreAssociatedComboBox.cpp \
    src/GlobalFunctions.cpp \
    src/Misc/SignalBlockerInterface.cpp \
    src/XUnsteadyBEM/FASTResult.cpp \
    src/XUnsteadyBEM/FastTwoDContextMenu.cpp \
    src/Graph/NewGraph.cpp \
    src/Graph/Axis.cpp \
    src/Graph/NewCurve.cpp \
    src/Graph/GraphOptionsDialog.cpp \
    src/Graph/ShowAsGraphInterface.cpp \
    src/QFem/QFEMTwoDContextMenu.cpp \
    src/QFem/forcingterm.cpp \
    src/QFem/staticequation.cpp \
    src/QFem/BladeStructureLoading.cpp \
    src/XBEM/ExportGeomDlg.cpp \
    src/TwoDContextMenu.cpp \
    src/XUnsteadyBEM/AboutFAST.cpp \
    src/XUnsteadyBEM/FASTMenu.cpp \
    src/Misc/FixedSizeLabel.cpp \
    src/XLLT/QLLTModule.cpp \
    src/XLLT/QLLTDock.cpp \
    src/XLLT/QLLTToolBar.cpp \
    src/QBladeApplication.cpp \
    src/XLLT/QLLTSimulation.cpp \
    src/VortexObjects/VortexNode.cpp \
    src/VortexObjects/VortexLine.cpp \
    src/XLLT/QLLTCreatorDialog.cpp \
    src/VortexObjects/DummyLine.cpp \
    src/XLLT/QLLTTwoDContextMenu.cpp \
    src/XLLT/QLLTCutPlane.cpp \
    src/XBEM/PolarSelectionDialog.cpp \
    src/XDMS/TurDmsModule.cpp \
    src/XDMS/TurDmsTurbineDock.cpp \
    src/XDMS/TurDmsSimulationDock.cpp \
    src/XDMS/TurDmsContextMenu.cpp \
    src/XDMS/TurDmsToolBar.cpp \
    src/SimulationDock.cpp \
    src/CreatorDock.cpp \
    src/XDMS/TurDmsTurbineCreatorDialog.cpp \
    src/SimulationModule.cpp \
    src/SimulationToolBar.cpp \
    src/XDMS/TurDmsSimulationCreatorDialog.cpp \
    src/ColorManager.cpp \
    src/Misc/LineStyleButton.cpp \
    src/Misc/LineStyleDialog.cpp \
    src/Misc/LineStyleComboBox.cpp \
    src/Misc/NewColorButton.cpp \
    src/TwoDGraphMenu.cpp \
    src/XUnsteadyBEM/WindFieldCreatorDialog.cpp \
    src/ParameterViewer.cpp \
    src/ParameterObject.cpp \
    src/ParameterGrid.cpp \
    src/TwoDWidgetInterface.cpp \
    src/XUnsteadyBEM/WindFieldDock.cpp \
    src/CreatorDialog.cpp \
    src/SimulationCreatorDialog.cpp \
    src/Noise/NoiseModule.cpp \
    src/Noise/NoiseSimulation.cpp \
    src/Noise/NoiseToolBar.cpp \
    src/Noise/NoiseDock.cpp \
    src/CreatorTwoDDock.cpp \
    src/Noise/NoiseCreatorDialog.cpp \
    src/Noise/NoiseOpPoint.cpp \
    src/Noise/NoiseCalculation.cpp \
    src/Noise/NoiseParameter.cpp \
    src/Noise/NoiseException.cpp \
    src/Noise/NoiseContextMenu.cpp \
    src/Noise/NoiseMenu.cpp \
    src/XDMS/StrutCreatorDialog.cpp \
    src/XDMS/RotDmsModule.cpp \
    src/XDMS/RotDmsToolBar.cpp \
    src/XDMS/RotDmsSimulationDock.cpp \
    src/XDMS/RotDmsContextMenu.cpp \
    src/XDMS/Strut.cpp \
    src/XDMS/RotDmsSimulationCreatorDialog.cpp \
    src/QBladeCMDApplication.cpp \
    src/XDMS/VawtDesignModule.cpp \
    src/XDMS/VawtDesignToolBar.cpp \
    src/XDMS/VawtDesignDock.cpp \
    src/GLMenu.cpp \
    src/XBEM/BladeWrapperModel.cpp \
    src/Misc/NumberEditDelegate.cpp \
    src/Misc/ComboBoxDelegate.cpp \
    src/Objects/CVectorf.cpp \
    src/StructModel/StrElem.cpp \
    src/StructModel/StrModel.cpp \
    src/StructModel/StrNode.cpp \
    src/Misc/MultiPolarDialog.cpp \
    src/Misc/MultiPolarDelegate.cpp \
    src/XDMS/NewStrutCreatorDialog.cpp \
    src/XDMS/BatchEditDialog.cpp \
    src/XDMS/CustomShapeDialog.cpp \
    IntegrationTest/integrationtest.cpp \
    src/XDMS/VawtDesignMenu.cpp \
    src/MultiSimulation/CommonMultiSimulationModule.cpp \
    src/MultiSimulation/BemMultiSimulationModule.cpp \
    src/MultiSimulation/CommonMultiSimulationDock.cpp \
    src/MultiSimulation/BemMultiSimulationDock.cpp \
    src/MultiSimulation/CommonMultiSimulationToolBar.cpp \
    src/MultiSimulation/BemMultiSimulationToolBar.cpp \
    src/MultiSimulation/CommonMultiSimulationCreatorDialog.cpp \
    src/MultiSimulation/BemMultiSimulationCreatorDialog.cpp \
    src/MultiSimulation/CommonMultiSimulationContextMenu.cpp \
    src/MultiSimulation/BemMultiSimulationContextMenu.cpp \
    src/MultiSimulation/DmsMultiSimulationContextMenu.cpp \
    src/MultiSimulation/DmsMultiSimulationModule.cpp \
    src/MultiSimulation/DmsMultiSimulationCreatorDialog.cpp \
    src/MultiSimulation/DmsMultiSimulationDock.cpp \
    src/MultiSimulation/DmsMultiSimulationToolBar.cpp \
    src/RotorSimulation/CommonRotorSimulationModule.cpp \
    src/RotorSimulation/CommonRotorSimulationToolBar.cpp \
    src/RotorSimulation/CommonRotorSimulationDock.cpp \
    src/RotorSimulation/CommonRotorSimulationCreatorDialog.cpp \
    src/RotorSimulation/CommonRotorSimulationContextMenu.cpp \
    src/RotorSimulation/BemRotorSimulationContextMenu.cpp \
    src/RotorSimulation/DmsRotorSimulationContextMenu.cpp \
    src/RotorSimulation/BemRotorSimulationDock.cpp \
    src/RotorSimulation/DmsRotorSimulationDock.cpp \
    src/RotorSimulation/BemRotorSimulationToolBar.cpp \
    src/RotorSimulation/DmsRotorSimulationToolBar.cpp \
    src/RotorSimulation/BemRotorSimulationModule.cpp \
    src/RotorSimulation/DmsRotorSimulationModule.cpp \
    src/RotorSimulation/BemRotorSimulationCreatorDialog.cpp \
    src/RotorSimulation/DmsRotorSimulationCreatorDialog.cpp \
    src/TurbineSimulation/CommonTurbineSimulationModule.cpp \
    src/TurbineSimulation/BemTurbineSimulationModule.cpp \
    src/TurbineSimulation/DmsTurbineSimulationModule.cpp \
    src/TurbineSimulation/CommonTurbineSimulationDock.cpp \
    src/TurbineSimulation/BemTurbineSimulationDock.cpp \
    src/TurbineSimulation/DmsTurbineSimulationDock.cpp \
    src/TurbineSimulation/CommonTurbineDock.cpp \
    src/TurbineSimulation/BemTurbineDock.cpp \
    src/TurbineSimulation/DmsTurbineDock.cpp \
    src/TurbineSimulation/CommonTurbineSimulationContextMenu.cpp \
    src/TurbineSimulation/BemTurbineSimulationContextMenu.cpp \
    src/TurbineSimulation/DmsTurbineSimulationContextMenu.cpp \
    src/TurbineSimulation/CommonTurbineSimulationToolBar.cpp \
    src/TurbineSimulation/BemTurbineSimulationToolBar.cpp \
    src/TurbineSimulation/DmsTurbineSimulationToolBar.cpp \
    src/TurbineSimulation/CommonTurbineSimulationCreatorDialog.cpp \
    src/TurbineSimulation/BemTurbineSimulationCreatorDialog.cpp \
    src/TurbineSimulation/DmsTurbineSimulationCreatorDialog.cpp \
    src/TurbineSimulation/CommonTurbineCreatorDialog.cpp \
    src/TurbineSimulation/BemTurbineCreatorDialog.cpp \
    src/TurbineSimulation/DmsTurbineCreatorDialog.cpp \
    src/XBEM/AFC.cpp \
    src/XBEM/DynPolarSet.cpp \
    src/XBEM/DynPolarSetDialog.cpp \
    src/XBEM/FlapCreatorDialog.cpp \
    src/VortexObjects/VortexParticle.cpp \
    src/StructModel/PID.cpp \
    src/QElast/QControl.cpp \
    src/QElast/QElast_Blade_Obj.cpp \
    src/QElast/QElast_CoordSys.cpp \
    src/QElast/QElast_HAWT_Turbine.cpp \
    src/QElast/QElast_Init_Turbine_Inputs.cpp \
    src/QElast/QElast_Params.cpp \
    src/QElast/QElast_RtHndSide.cpp \
    src/QElast/QElast_Tower_Obj.cpp \
    src/QElast/QElast_Turbine_Inputs.cpp \
    src/StructModel/StrModelDock.cpp \
    src/StructModel/CoordSys.cpp \
    src/IceThrowSimulation/IceThrowSimulation.cpp \
    src/IceThrowSimulation/Particle.cpp \
    src/KIFMM/Kernels/Kernel_2D_Vortex_Particle.cpp \
    src/KIFMM/Kernels/Kernel_3D_Dipol.cpp \
    src/KIFMM/Kernels/Kernel_3D_Potential_Dipol.cpp \
    src/KIFMM/Kernels/Kernel_3D_Potential_Source.cpp \
    src/KIFMM/Kernels/Kernel_3D_Source.cpp \
    src/KIFMM/Kernels/Kernel_3D_Vortex_Filament.cpp \
    src/KIFMM/Kernels/Kernel_3D_Vortex_Particle.cpp \
    src/KIFMM/Testing/KIFMM_Filament_Test.cpp \
    src/KIFMM/Testing/KIFMM_Test.cpp \
    src/KIFMM/KIFMM.cpp \
    src/KIFMM/KIFMM_Box.cpp \
    src/KIFMM/KIFMM_Domain.cpp \
    src/KIFMM/KIFMM_Interp.cpp \
    src/KIFMM/KIFMM_Node.cpp \
    src/KIFMM/KIFMM_OpenCL.cpp \
    src/QMultiLLT/QMultiSimDock.cpp \
    src/QMultiLLT/QMultiSimToolBar.cpp \
    src/QMultiLLT/QMultiSimModule.cpp \
    src/QMultiLLT/QMultiSimTwoDContextMenu.cpp \
    src/QTurbine/QTurbineModule.cpp \
    src/QTurbine/QTurbineDock.cpp \
    src/QTurbine/QTurbineToolBar.cpp \
    src/QTurbine/QTurbine.cpp \
    src/QTurbine/QTurbineCreatorDialog.cpp \
    src/QTurbine/QTurbineTwoDContextMenu.cpp \
    src/QMultiLLT/QMultiSimulation.cpp \
    src/QMultiLLT/QMultiSimulationCreatorDialog.cpp \
    src/StructModel/StrModelMulti.cpp \
    src/StructModel/StrObjects.cpp \
    src/QTurbine/QTurbineSimulationData.cpp \
    src/QTurbine/QTurbineResults.cpp \
    src/QTurbine/QTurbineGlRendering.cpp \
    src/OpenCLSetup.cpp \
    src/FlightSimulator/Plane.cpp \
    src/FlightSimulator/PlaneDesignerDock.cpp \
    src/FlightSimulator/PlaneDesignerModule.cpp \
    src/FlightSimulator/PlaneDesignerToolbar.cpp \
    src/FlightSimulator/QFlightCreatorDialog.cpp \
    src/FlightSimulator/QFlightDock.cpp \
    src/FlightSimulator/QFlightModule.cpp \
    src/FlightSimulator/QFlightSimCreatorDialog.cpp \
    src/FlightSimulator/QFlightSimulation.cpp \
    src/FlightSimulator/QFlightStructuralModel.cpp \
    src/FlightSimulator/QFlightToolBar.cpp \
    src/FlightSimulator/QFlightTwoDContextMenu.cpp \
    src/FlightSimulator/WingDelegate.cpp \
    src/FlightSimulator/WingDesignerDock.cpp \
    src/FlightSimulator/WingDesignerModule.cpp \
    src/FlightSimulator/WingDesignerToolbar.cpp \
    src/InterfaceDll/QBladeDLLApplication.cpp \
    src/InterfaceDll/QBladeDLLInclude.cpp


HEADERS += src/MainFrame.h \
    src/Params.h \
    src/Globals.h \
    src/TwoDWidget.h \
    src/GLWidget.h \
    src/Misc/GLLightDlg.h \
    src/XDirect/XFoil.h \
    src/XDirect/XFoilAnalysisDlg.h \
    src/XDirect/XFoilAdvancedDlg.h \
    src/XDirect/XDirect.h \
    src/XDirect/TwoDPanelDlg.h \
    src/XDirect/TEGapDlg.h \
    src/XDirect/InterpolateFoilsDlg.h \
    src/XDirect/FoilGeomDlg.h \
    src/XDirect/FoilCoordDlg.h \
    src/XDirect/ReListDlg.h \
    src/XDirect/XDirectStyleDlg.h \
    src/XDirect/ManageFoilsDlg.h \
    src/XDirect/NacaFoilDlg.h \
    src/XDirect/LEDlg.h \
    src/XDirect/FoilPolarDlg.h \
    src/XDirect/FlapDlg.h \
    src/XDirect/CAddDlg.h \
    src/XDirect/BatchDlg.h \
    src/XDirect/BatchThreadDlg.h \
    src/XDirect/XFoilTask.h \
    src/XInverse/XInverse.h \
    src/XInverse/InverseOptionsDlg.h \
    src/XInverse/FoilSelectionDlg.h \
    src/XInverse/PertDlg.h \
    src/Objects/Surface.h \
    src/Objects/Spline.h \
    src/Objects/Sf.h \
    src/Objects/OpPoint.h \
    src/Objects/Quaternion.h \
    src/Objects/Polar.h \
    src/Objects/CVector.h \
    src/VortexObjects/Panel.h \
    src/Objects/Foil.h \
    src/Misc/PolarFilterDlg.h \
    src/Misc/TranslatorDlg.h \
    src/Misc/UnitsDlg.h \
    src/Misc/LinePickerDlg.h \
    src/Misc/LineDelegate.h \
    src/Misc/FloatEditDelegate.h \
    src/Misc/DisplaySettingsDlg.h \
    src/Misc/ColorButton.h \
    src/Misc/LineCbBox.h \
    src/Misc/LineButton.h \
    src/Misc/EditPlrDlg.h \
    src/Misc/AboutQ5.h \
    src/Misc/ObjectPropsDlg.h \
    src/Graph/GraphWidget.h \
    src/Graph/Graph.h \
    src/Graph/GraphDlg.h \
    src/Graph/Curve.h \
    src/Graph/QGraph.h \
    src/Design/AFoil.h \
    src/Design/AFoilGridDlg.h \
    src/Design/LECircleDlg.h \
    src/Design/SplineCtrlsDlg.h \
    src/Design/FoilTableDelegate.h \
    src/Design/AFoilTableDlg.h \
    src/XBEM/TData.h \
    src/XBEM/TBEMData.h \
    src/XBEM/SimuWidget.h \
    src/XBEM/OptimizeDlg.h \
    src/XBEM/Edit360PolarDlg.h \
    src/XBEM/CreateBEMDlg.h \
    src/XBEM/BladeScaleDlg.h \
    src/XBEM/BEMData.h \
    src/XBEM/BEM.h \
    src/XBEM/BData.h \
    src/XBEM/AboutBEM.h \
    src/XBEM/Blade.h \
    src/XBEM/BladeDelegate.h \
    src/XBEM/BladeAxisDelegate.h \
    src/XBEM/CBEMData.h \
    src/XBEM/PrescribedValuesDlg.h \
    src/XBEM/CircularFoilDlg.h \
    src/XBEM/BEMSimDock.h \
    src/XBEM/BEMDock.h \
    src/XDMS/DMS.h \
    src/XDMS/SimuWidgetDMS.h \
    src/XDMS/BladeDelegate2.h \
    src/XDMS/OptimizeDlgVAWT.h \
    src/XDMS/BladeScaleDlgVAWT.h \
    src/XDMS/CreateDMSDlg.h \
    src/XDMS/DMSData.h \
    src/XDMS/DData.h \
    src/XDMS/TDMSData.h \
    src/XDMS/DMSSimDock.h \
    src/XDMS/DMSDock.h \
    src/XDMS/CDMSData.h \
    src/XUnsteadyBEM/WindField.h \
    src/XWidgets.h \
    src/XUnsteadyBEM/WindFieldModule.h \
    src/Module.h \
    src/XUnsteadyBEM/WindFieldToolBar.h \
    src/ScrolledDock.h \
    src/Store.h \
    src/XUnsteadyBEM/WindFieldMenu.h \
    src/QFem/taperedelem.h \
    src/QFem/structintegrator.h \
    src/QFem/structelem.h \
    src/QFem/node.h \
    src/QFem/eqnmotion.h \
    src/QFem/clipper.cpp \
    src/QFem/unitvector.h \
    src/QFem/mode.h \
    src/QFem/deformationvector.h \
    src/QFem/QFEMDock.h \
    src/QFem/QFEMToolBar.h \
    src/QFem/QFEMModule.h \
    src/QFem/QFEMMenu.h \
    src/QFem/BladeStructure.h \
    src/QFem/StructDelegate.h \
    src/StorableObject.h \
    src/XUnsteadyBEM/FASTModule.h \
    src/XUnsteadyBEM/FASTDock.h \
    src/XUnsteadyBEM/FASTSimulation.h \
    src/XUnsteadyBEM/FASTSimulationCreatorDialog.h \
    src/XUnsteadyBEM/FASTSimulationToolBar.h \
    src/XUnsteadyBEM/FASTSimulationProgressDialog.h \
    src/XBEM/BEMToolbar.h \
    src/XDMS/DMSToolbar.h \
    src/XBEM/360Polar.h \
    src/Misc/NumberEdit.h \
    src/Serializer.h \
    src/StoreAssociatedComboBox.h \
    src/StoreAssociatedComboBox_include.h \
    src/Store_include.h \
    src/StorableObject_heirs.h \
    src/GlobalFunctions.h \
    src/Misc/SignalBlockerInterface.h \
    src/XUnsteadyBEM/FASTResult.h \
    src/XUnsteadyBEM/FastTwoDContextMenu.h \
    src/Graph/NewGraph.h \
    src/Graph/Axis.h \
    src/Graph/NewCurve.h \
    src/Graph/GraphOptionsDialog.h \
    src/Graph/ShowAsGraphInterface.h \
    src/QFem/QFEMTwoDContextMenu.h \
    src/QFem/forcingterm.h \
    src/QFem/staticequation.h \
    src/QFem/BladeStructureLoading.h \
    src/XBEM/ExportGeomDlg.h \
    src/TwoDContextMenu.h \
    src/XUnsteadyBEM/AboutFAST.h \
    src/XUnsteadyBEM/FASTMenu.h \
    src/Misc/FixedSizeLabel.h \
    src/XLLT/QLLTModule.h \
    src/XLLT/QLLTDock.h \
    src/XLLT/QLLTToolBar.h \
    src/QBladeApplication.h \
    src/XLLT/QLLTSimulation.h \
    src/VortexObjects/VortexNode.h \
    src/VortexObjects/VortexLine.h \
    src/XLLT/QLLTCreatorDialog.h \
    src/VortexObjects/DummyLine.h \
    src/XLLT/QLLTTwoDContextMenu.h \
    src/XLLT/QLLTCutPlane.h \
    src/XBEM/PolarSelectionDialog.h \
    src/XDMS/TurDmsModule.h \
    src/XDMS/TurDmsTurbineDock.h \
    src/XDMS/TurDmsSimulationDock.h \
    src/XDMS/TurDmsContextMenu.h \
    src/XDMS/TurDmsToolBar.h \
    src/SimulationDock.h \
    src/CreatorDock.h \
    src/XDMS/TurDmsTurbineCreatorDialog.h \
    src/SimulationModule.h \
    src/SimulationToolBar.h \
    src/XDMS/TurDmsSimulationCreatorDialog.h \
    src/ColorManager.h \
    src/Misc/LineStyleButton.h \
    src/Misc/LineStyleDialog.h \
    src/Misc/LineStyleComboBox.h \
    src/Misc/NewColorButton.h \
    src/TwoDGraphMenu.h \
    src/XUnsteadyBEM/WindFieldCreatorDialog.h \
    src/ParameterViewer.h \
    src/ParameterObject.h \
    src/ParameterGrid.h \
    src/ParameterKeys.h \
    src/TwoDWidgetInterface.h \
    src/XUnsteadyBEM/WindFieldDock.h \
    src/CreatorDialog.h \
    src/SimulationCreatorDialog.h \
    src/Noise/NoiseModule.h \
    src/Noise/NoiseSimulation.h \
    src/Noise/NoiseToolBar.h \
    src/Noise/NoiseDock.h \
    src/CreatorTwoDDock.h \
    src/Noise/NoiseCreatorDialog.h \
    src/Noise/NoiseOpPoint.h \
    src/Noise/NoiseCalculation.h \
    src/Noise/NoiseParameter.h \
    src/Noise/NoiseException.h \
    src/Noise/NoiseContextMenu.h \
    src/Noise/NoiseMenu.h \
    src/XDMS/StrutCreatorDialog.h \
    src/XDMS/RotDmsModule.h \
    src/XDMS/RotDmsToolBar.h \
    src/XDMS/RotDmsSimulationDock.h \
    src/XDMS/RotDmsContextMenu.h \
    src/XDMS/Strut.h \
    src/XDMS/RotDmsSimulationCreatorDialog.h \
    src/QBladeCMDApplication.h \
    src/XDMS/VawtDesignModule.h \
    src/XDMS/VawtDesignToolBar.h \
    src/XDMS/VawtDesignDock.h \
    src/GLMenu.h \
    src/XBEM/BladeWrapperModel.h \
    src/Misc/NumberEditDelegate.h \
    src/Misc/ComboBoxDelegate.h \
    src/Objects/CVectorf.h \
    src/StructModel/StrElem.h \
    src/StructModel/StrModel.h \
    src/StructModel/StrNode.h \
    src/Misc/MultiPolarDialog.h \
    src/Misc/MultiPolarDelegate.h \
    src/XDMS/NewStrutCreatorDialog.h \
    src/XDMS/BatchEditDialog.h \
    src/XDMS/CustomShapeDialog.h \
    IntegrationTest/integrationtest.h \
    src/CompileSettings.h \
    src/XDMS/VawtDesignMenu.h \
    src/MultiSimulation/CommonMultiSimulationModule.h \
    src/MultiSimulation/BemMultiSimulationModule.h \
    src/MultiSimulation/CommonMultiSimulationDock.h \
    src/MultiSimulation/BemMultiSimulationDock.h \
    src/MultiSimulation/CommonMultiSimulationToolBar.h \
    src/MultiSimulation/BemMultiSimulationToolBar.h \
    src/MultiSimulation/CommonMultiSimulationCreatorDialog.h \
    src/MultiSimulation/BemMultiSimulationCreatorDialog.h \
    src/MultiSimulation/CommonMultiSimulationContextMenu.h \
    src/MultiSimulation/BemMultiSimulationContextMenu.h \
    src/MultiSimulation/DmsMultiSimulationContextMenu.h \
    src/MultiSimulation/DmsMultiSimulationModule.h \
    src/MultiSimulation/DmsMultiSimulationCreatorDialog.h \
    src/MultiSimulation/DmsMultiSimulationDock.h \
    src/MultiSimulation/DmsMultiSimulationToolBar.h \
    src/RotorSimulation/CommonRotorSimulationModule.h \
    src/RotorSimulation/CommonRotorSimulationToolBar.h \
    src/RotorSimulation/CommonRotorSimulationDock.h \
    src/RotorSimulation/CommonRotorSimulationCreatorDialog.h \
    src/RotorSimulation/CommonRotorSimulationContextMenu.h \
    src/RotorSimulation/BemRotorSimulationContextMenu.h \
    src/RotorSimulation/DmsRotorSimulationContextMenu.h \
    src/RotorSimulation/BemRotorSimulationDock.h \
    src/RotorSimulation/DmsRotorSimulationDock.h \
    src/RotorSimulation/BemRotorSimulationToolBar.h \
    src/RotorSimulation/DmsRotorSimulationToolBar.h \
    src/RotorSimulation/BemRotorSimulationModule.h \
    src/RotorSimulation/DmsRotorSimulationModule.h \
    src/RotorSimulation/BemRotorSimulationCreatorDialog.h \
    src/RotorSimulation/DmsRotorSimulationCreatorDialog.h \
    src/TurbineSimulation/CommonTurbineSimulationModule.h \
    src/TurbineSimulation/BemTurbineSimulationModule.h \
    src/TurbineSimulation/DmsTurbineSimulationModule.h \
    src/TurbineSimulation/CommonTurbineSimulationDock.h \
    src/TurbineSimulation/BemTurbineSimulationDock.h \
    src/TurbineSimulation/DmsTurbineSimulationDock.h \
    src/TurbineSimulation/CommonTurbineDock.h \
    src/TurbineSimulation/BemTurbineDock.h \
    src/TurbineSimulation/DmsTurbineDock.h \
    src/TurbineSimulation/CommonTurbineSimulationContextMenu.h \
    src/TurbineSimulation/BemTurbineSimulationContextMenu.h \
    src/TurbineSimulation/DmsTurbineSimulationContextMenu.h \
    src/TurbineSimulation/CommonTurbineSimulationToolBar.h \
    src/TurbineSimulation/BemTurbineSimulationToolBar.h \
    src/TurbineSimulation/DmsTurbineSimulationToolBar.h \
    src/TurbineSimulation/CommonTurbineSimulationCreatorDialog.h \
    src/TurbineSimulation/BemTurbineSimulationCreatorDialog.h \
    src/TurbineSimulation/DmsTurbineSimulationCreatorDialog.h \
    src/TurbineSimulation/CommonTurbineCreatorDialog.h \
    src/TurbineSimulation/BemTurbineCreatorDialog.h \
    src/TurbineSimulation/DmsTurbineCreatorDialog.h \
    src/XBEM/AFC.h \
    src/XBEM/DynPolarSet.h \
    src/XBEM/DynPolarSetDialog.h \
    src/XBEM/FlapCreatorDialog.h \
    src/VortexObjects/VortexParticle.h \
    src/StructModel/PID.h \
    src/QElast/QControl.h \
    src/QElast/QElast_Blade_Obj.h \
    src/QElast/QElast_CoordSys.h \
    src/QElast/QElast_HAWT_Turbine.h \
    src/QElast/QElast_Init_Turbine_Inputs.h \
    src/QElast/QElast_Params.h \
    src/QElast/QElast_RtHndSide.h \
    src/QElast/QElast_Tower_Obj.h \
    src/QElast/QElast_Turbine_Inputs.h \
    src/StructModel/StrModelDock.h \
    src/StructModel/CoordSys.h \
    src/IceThrowSimulation/IceThrowSimulation.h \
    src/IceThrowSimulation/Particle.h \
    src/QElast/QElast_Types.h \
    src/KIFMM/Kernels/Kernel.h \
    src/KIFMM/Kernels/Kernel_2D_Vortex_Particle.h \
    src/KIFMM/Kernels/Kernel_3D_Dipol.h \
    src/KIFMM/Kernels/Kernel_3D_Potential_Dipol.h \
    src/KIFMM/Kernels/Kernel_3D_Potential_Source.h \
    src/KIFMM/Kernels/Kernel_3D_Source.h \
    src/KIFMM/Kernels/Kernel_3D_Vortex_Filament.h \
    src/KIFMM/Kernels/Kernel_3D_Vortex_Particle.h \
    src/KIFMM/Testing/KIFMM_Filament_Test.h \
    src/KIFMM/Testing/KIFMM_Test.h \
    src/KIFMM/KIFMM.h \
    src/KIFMM/KIFMM_Box.h \
    src/KIFMM/KIFMM_Domain.h \
    src/KIFMM/KIFMM_Index.h \
    src/KIFMM/KIFMM_Input.h \
    src/KIFMM/KIFMM_Interp.h \
    src/KIFMM/KIFMM_Kernels.h \
    src/KIFMM/KIFMM_Node.h \
    src/KIFMM/KIFMM_OpenCL.h \
    src/KIFMM/KIFMM_Types.h \
    src/QMultiLLT/QMultiSimDock.h \
    src/QMultiLLT/QMultiSimToolBar.h \
    src/QMultiLLT/QMultiSimModule.h \
    src/QMultiLLT/QMultiSimTwoDContextMenu.h \
    src/QTurbine/QTurbineModule.h \
    src/QTurbine/QTurbineDock.h \
    src/QTurbine/QTurbineToolBar.h \
    src/QTurbine/QTurbine.h \
    src/QTurbine/QTurbineCreatorDialog.h \
    src/QTurbine/QTurbineTwoDContextMenu.h \
    src/QMultiLLT/QMultiSimulation.h \
    src/QMultiLLT/QMultiSimulationCreatorDialog.h \
    src/StructModel/StrModelMulti.h \
    src/StructModel/StrObjects.h \
    src/QTurbine/QTurbineSimulationData.h \
    src/QTurbine/QTurbineResults.h \
    src/QTurbine/QTurbineGlRendering.h \
    src/QMultiLLT/QmultiSimulationThread.h \
    src/OpenCLSetup.h \
    src/FlightSimulator/Plane.h \
    src/FlightSimulator/PlaneDesignerDock.h \
    src/FlightSimulator/PlaneDesignerModule.h \
    src/FlightSimulator/PlaneDesignerToolbar.h \
    src/FlightSimulator/QFlightCreatorDialog.h \
    src/FlightSimulator/QFlightDock.h \
    src/FlightSimulator/QFlightModule.h \
    src/FlightSimulator/QFlightSimCreatorDialog.h \
    src/FlightSimulator/QFlightSimulation.h \
    src/FlightSimulator/QFlightStructuralModel.h \
    src/FlightSimulator/QFlightToolBar.h \
    src/FlightSimulator/QFlightTwoDContextMenu.h \
    src/FlightSimulator/WingDelegate.h \
    src/FlightSimulator/WingDesignerDock.h \
    src/FlightSimulator/WingDesignerModule.h \
    src/FlightSimulator/WingDesignerToolbar.h \
    src/InterfaceDll/QBladeDLLApplication.h \
    src/InterfaceDll/QBladeDLLInclude.h