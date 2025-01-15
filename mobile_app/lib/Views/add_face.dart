import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class AddFacePage extends StatefulWidget {
  const AddFacePage({super.key});

  @override
  AddFacePageState createState() => AddFacePageState();  
}

class AddFacePageState extends State<AddFacePage> {
  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  bool _isCameraReady = false;
  bool _isInitializing = true; 

  // Initialize the camera
  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      final camera = _cameras.first;

      _cameraController = CameraController(camera, ResolutionPreset.high);
      await _cameraController.initialize();
      if (!mounted) return; 
      setState(() {
        _isCameraReady = true;
        _isInitializing = false;
      });
    } catch (e) {
      
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
      });
      // print('Error initializing camera: $e');
    }
  }

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Camera Preview")),
      body: _isInitializing
          ? const Center(child: CircularProgressIndicator()) 
          : _isCameraReady
              ? CameraPreview(_cameraController)
              : const Center(child: Text('Failed to initialize the camera')),
    );
  }
}
