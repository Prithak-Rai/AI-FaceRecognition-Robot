import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:app/SQLite/database_helper.dart';
import 'package:app/Views/home.dart';
import 'package:app/Views/profile.dart';
import 'package:app/Views/settings.dart';
import 'package:app/Views/faces.dart';
import 'package:app/JSON/users.dart';
import 'package:app/Components/button.dart';

class AddFacePage extends StatefulWidget {
  final Users? profile;
  final int? userId;
  
  const AddFacePage({super.key, this.profile, this.userId});

  @override
  AddFacePageState createState() => AddFacePageState();  
}

class AddFacePageState extends State<AddFacePage> {
  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  bool _isCameraReady = false;
  bool _isInitializing = true; 
  bool _isProcessing = false;
  int _selectedIndex = 1; // Set to 1 since we're in the Faces section

  // Initialize the camera
  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      
      // Check if front camera exists and use it
      CameraDescription selectedCamera = _cameras.first;
      for (var camera in _cameras) {
        if (camera.lensDirection == CameraLensDirection.front) {
          selectedCamera = camera;
          break;
        }
      }

      _cameraController = CameraController(selectedCamera, ResolutionPreset.high);
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
      print('Error initializing camera: $e');
    }
  }

  Future<String?> _showNameDialog(BuildContext context) async {
    final TextEditingController nameController = TextEditingController();
    return showDialog<String>(
      context: context,
      barrierDismissible: false, // User must tap button to close dialog
      builder: (context) => AlertDialog(
        title: const Text('Enter Face Name'),
        content: TextField(
          controller: nameController,
          decoration: const InputDecoration(
            labelText: 'Name',
            hintText: 'Enter a name for this face'
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, nameController.text),
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  void _onItemTapped(int index) {
    if (index == _selectedIndex) return;
    
    if (index == 0) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomePage(profile: widget.profile)),
      );
    } else if (index == 1) {
      // We're already in the Faces section, but navigate to the main Faces page
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const FacesScreen(),
        ),
      );
    } else if (index == 2) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => Profile(profile: widget.profile)),
      );
    } else if (index == 3) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const SettingsPage()),
      );
    }
  }

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    if (_cameraController.value.isInitialized) {
      _cameraController.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topRight,
          end: Alignment.bottomLeft,
          colors: [
            Colors.blue,
            Color(0xFFFFA2A2),
          ],
        ),
      ),
      child: Scaffold(
        backgroundColor: Colors.transparent,
        appBar: AppBar(
          title: const Text(
            "Add New Face",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          ),
          backgroundColor: Colors.transparent,
          elevation: 0,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back, color: Colors.white),
            onPressed: () {
              Navigator.pop(context);
            },
          ),
        ),
        body: _isInitializing
            ? const Center(child: CircularProgressIndicator(color: Colors.white)) 
            : !_isCameraReady
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.camera_alt_outlined,
                          color: Colors.white,
                          size: 80,
                        ),
                        const SizedBox(height: 20),
                        const Text(
                          'Camera not available',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          'Please allow camera access to register faces',
                          style: TextStyle(color: Colors.white70),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 30),
                        Button(
                          label: "Try Again",
                          press: () {
                            setState(() {
                              _isInitializing = true;
                            });
                            _initializeCamera();
                          },
                        ),
                      ],
                    ),
                  )
                : Column(
                    children: [
                      Expanded(
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            // Camera preview
                            ClipRRect(
                              borderRadius: BorderRadius.circular(12),
                              child: CameraPreview(_cameraController),
                            ),
                            
                            // Face outline guide
                            Container(
                              width: 220,
                              height: 220,
                              decoration: BoxDecoration(
                                border: Border.all(color: Colors.white, width: 2),
                                shape: BoxShape.circle,
                              ),
                            ),
                            
                            // Processing indicator
                            if (_isProcessing)
                              Container(
                                color: Colors.black45,
                                child: const Center(
                                  child: CircularProgressIndicator(color: Colors.white),
                                ),
                              ),
                          ],
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(30.0),
                        child: Column(
                          children: [
                            const Text(
                              'Position your face within the circle',
                              style: TextStyle(color: Colors.white, fontSize: 16),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 20),
                            ElevatedButton(
                              onPressed: _isProcessing 
                                  ? null 
                                  : () async {
                                      setState(() {
                                        _isProcessing = true;
                                      });
                                      
                                      try {
                                        // Capture the image
                                        final XFile imageFile = await _cameraController.takePicture();
                                        
                                        // Read image as bytes
                                        final imageBytes = await imageFile.readAsBytes();
                                        
                                        // Wait a moment to show "processing" state
                                        await Future.delayed(const Duration(milliseconds: 500));
                                        
                                        if (!mounted) return;
                                        
                                        // Show dialog to enter face name
                                        final String? faceName = await _showNameDialog(context);
                                        
                                        if (faceName != null && faceName.isNotEmpty) {
                                          // Save face to database
                                          final databaseHelper = DatabaseHelper();
                                          final userId = widget.userId ?? widget.profile?.usrId;
                                          
                                          if (userId != null) {
                                            await databaseHelper.addFace(faceName, imageBytes, userId);
                                            
                                            // Show success message and navigate back
                                            if (!mounted) return;
                                            ScaffoldMessenger.of(context).showSnackBar(
                                              const SnackBar(content: Text('Face added successfully')),
                                            );
                                            Navigator.pop(context); // Return to faces page
                                          } else {
                                            // Show error if user ID is missing
                                            ScaffoldMessenger.of(context).showSnackBar(
                                              const SnackBar(content: Text('Error: User information missing')),
                                            );
                                          }
                                        }
                                      } catch (e) {
                                        print('Error capturing image: $e');
                                        if (!mounted) return;
                                        ScaffoldMessenger.of(context).showSnackBar(
                                          SnackBar(content: Text('Error: $e')),
                                        );
                                      } finally {
                                        if (mounted) {
                                          setState(() {
                                            _isProcessing = false;
                                          });
                                        }
                                      }
                                    },
                              style: ElevatedButton.styleFrom(
                                shape: const CircleBorder(),
                                padding: const EdgeInsets.all(20),
                                backgroundColor: Colors.white,
                                disabledBackgroundColor: Colors.grey,
                              ),
                              child: const Icon(Icons.camera_alt, color: Colors.blue, size: 36),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
        extendBody: true, // Important - allows content to render behind the navigation bar
        bottomNavigationBar: Theme(
          data: Theme.of(context).copyWith(
            canvasColor: Colors.transparent,
          ),
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.transparent,
                  Colors.black.withOpacity(0.2),
                ],
              ),
            ),
            child: BottomNavigationBar(
              backgroundColor: Colors.transparent,
              elevation: 0,
              selectedItemColor: Colors.white,
              unselectedItemColor: Colors.white70,
              currentIndex: _selectedIndex,
              onTap: _onItemTapped,
              type: BottomNavigationBarType.fixed,
              items: const [
                BottomNavigationBarItem(
                  icon: Icon(Icons.home),
                  label: 'Home',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.face),
                  label: 'Faces',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.person),
                  label: 'Profile',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.settings),
                  label: 'Settings',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}