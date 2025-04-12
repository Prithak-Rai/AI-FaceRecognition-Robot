import 'package:flutter/material.dart';
import 'package:flutter_sqlite_auth_app/Views/add_face.dart';
import 'package:flutter_sqlite_auth_app/Components/button.dart';
import 'package:flutter_sqlite_auth_app/Views/profile.dart';
import 'package:flutter_sqlite_auth_app/JSON/users.dart';
import 'package:flutter_sqlite_auth_app/Views/settings.dart';
import 'package:flutter_sqlite_auth_app/Views/faces.dart';

class HomePage extends StatefulWidget {
  final Users? profile;

  const HomePage({Key? key, this.profile}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;
  bool _isConnected = true; // Replace with actual connection check logic

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });

    if (index == 1) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => FacesPage(profile: widget.profile), // Pass profile data
        ),
      );
    } else if (index == 2) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => Profile(profile: widget.profile),
        ),
      );
    } else if (index == 3) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => const SettingsPage(),
        ),
      );
    }
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
        body: SingleChildScrollView(
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 20.0, horizontal: 20),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // System status indicator
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _isConnected ? Icons.wifi : Icons.wifi_off,
                          color: _isConnected ? Colors.green : Colors.red,
                          size: 16,
                        ),
                        const SizedBox(width: 5),
                        Text(
                          _isConnected ? "System Connected" : "System Offline",
                          style: const TextStyle(color: Colors.white, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    'Welcome to FaceBot',
                    style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  const SizedBox(height: 10),
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: Text(
                      'Smart facial recognition for your Raspberry Pi robot',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ),
                  const SizedBox(height: 30),
                  
                  // Tutorial card
                  Container(
                    padding: const EdgeInsets.all(15),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Icon(Icons.lightbulb_outline, color: Colors.yellow[100], size: 24),
                            const SizedBox(width: 10),
                            const Text(
                              "How it works",
                              style: TextStyle(
                                fontSize: 18, 
                                fontWeight: FontWeight.bold,
                                color: Colors.white
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        const Row(
                          children: [
                            CircleAvatar(
                              backgroundColor: Colors.white38,
                              radius: 12,
                              child: Text("1", style: TextStyle(color: Colors.white)),
                            ),
                            SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                "Add your face to the system",
                                style: TextStyle(color: Colors.white),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        const Row(
                          children: [
                            CircleAvatar(
                              backgroundColor: Colors.white38,
                              radius: 12,
                              child: Text("2", style: TextStyle(color: Colors.white)),
                            ),
                            SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                "Your robot will recognize you automatically",
                                style: TextStyle(color: Colors.white),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        const Row(
                          children: [
                            CircleAvatar(
                              backgroundColor: Colors.white38,
                              radius: 12,
                              child: Text("3", style: TextStyle(color: Colors.white)),
                            ),
                            SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                "Customize actions in settings",
                                style: TextStyle(color: Colors.white),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 30),
                  
                  // Main action button
                  Button(
                    label: "Add Your Face",
                    press: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const AddFacePage()),
                      );
                    },
                  ),
                  
                  const SizedBox(height: 15),
                  
                  // Alternative authentication option
                  TextButton(
                    onPressed: () {
                      // Navigate to alternative auth page
                    },
                    child: const Text(
                      "Use alternative login method",
                      style: TextStyle(color: Colors.white70),
                    ),
                  ),
                  
                  const SizedBox(height: 20),
                  
                  // Privacy notice
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.security, color: Colors.white70, size: 18),
                        SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            "Your facial data is stored locally on your device and never shared.",
                            style: TextStyle(color: Colors.white70, fontSize: 12),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
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
              type: BottomNavigationBarType.fixed, // Add this to support 4 items
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