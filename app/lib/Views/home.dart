import 'package:flutter/material.dart';
import 'package:app/Views/add_face.dart';
import 'package:app/Components/button.dart';
import 'package:app/Views/profile.dart';
import 'package:app/JSON/users.dart';
import 'package:app/Views/settings.dart';
// import 'package:app/Views/faces.dart';
import 'package:app/Views/notification.dart';
import 'package:provider/provider.dart';
import 'package:app/services/mqtt_service.dart';

class HomePage extends StatefulWidget {
  final Users? profile;

  const HomePage({Key? key, this.profile}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with WidgetsBindingObserver {
  int _selectedIndex = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final mqttService = Provider.of<MQTTService>(context, listen: false);
      if (!mqttService.isConnected) {
        mqttService.initialize();
      }
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // Refresh MQTT connection when app comes to foreground
      final mqttService = Provider.of<MQTTService>(context, listen: false);
      if (!mqttService.isConnected) {
        mqttService.initialize();
      }
    }
  }

  // Add method to get status color
  Color _getStatusColor(bool isRobotConnected) {
    return isRobotConnected ? Colors.green : Colors.red;
  }

  // Add method to get status text
  String _getStatusText(bool isRobotConnected) {
    return isRobotConnected ? "Robot Connected" : "Robot Offline";
  }

  // Add method to get status icon
  IconData _getStatusIcon(bool isRobotConnected) {
    return isRobotConnected ? Icons.wifi : Icons.wifi_off;
  }

  // Add method to get status message
  String _getStatusMessage(bool isRobotConnected) {
    return isRobotConnected 
        ? "Your robot is online and ready"
        : "Waiting for robot connection...";
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });

    // if (index == 1) {
    //   Navigator.push(
    //     context,
    //     MaterialPageRoute(
    //       builder: (context) => const FacesScreen(),
    //     ),
    //   );
    // } else 
    if (index == 1) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => Profile(profile: widget.profile),
        ),
      );
    } else if (index == 2) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => const SettingsPage(),
        ),
      );
    }
  }

  // Method to navigate to notifications screen
  void _navigateToNotifications() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const NotificationsScreen(),
      ),
    );
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
          backgroundColor: Colors.transparent,
          elevation: 0,
          title: const Text(
            'FaceBot',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          ),
          actions: [
            // Notification button with dynamic badge
            Consumer<MQTTService>(
              builder: (context, mqttService, child) {
                final notificationCount = mqttService.notifications.length;
                
                return Stack(
                  alignment: Alignment.center,
                  children: [
                    // IconButton(
                    //   icon: const Icon(Icons.notifications_outlined, color: Colors.white),
                    //   onPressed: _navigateToNotifications,
                    // ),
                    // Dynamic notification badge
                    if (notificationCount > 0)
                      Positioned(
                        top: 8,
                        right: 8,
                        child: Container(
                          padding: const EdgeInsets.all(2),
                          decoration: const BoxDecoration(
                            color: Colors.red,
                            shape: BoxShape.circle,
                          ),
                          constraints: const BoxConstraints(
                            minWidth: 16,
                            minHeight: 16,
                          ),
                          child: Text(
                            notificationCount > 99 ? '99+' : notificationCount.toString(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ),
                  ],
                );
              },
            ),
          ],
        ),
        body: Consumer<MQTTService>(
          builder: (context, mqttService, child) {
            final bool isRobotConnected = mqttService.isRobotConnected;
            final bool isMqttConnected = mqttService.isConnected;
            
            return SingleChildScrollView(
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 20.0, horizontal: 20),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // System status indicator with animation
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 500),
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: _getStatusColor(isRobotConnected).withOpacity(0.5),
                            width: 1,
                          ),
                        ),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            // Robot status
                            Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                AnimatedSwitcher(
                                  duration: const Duration(milliseconds: 300),
                                  child: Icon(
                                    _getStatusIcon(isRobotConnected),
                                    color: _getStatusColor(isRobotConnected),
                                    size: 16,
                                    key: ValueKey(isRobotConnected),
                                  ),
                                ),
                                const SizedBox(width: 5),
                                AnimatedDefaultTextStyle(
                                  duration: const Duration(milliseconds: 300),
                                  style: TextStyle(
                                    color: _getStatusColor(isRobotConnected),
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                  ),
                                  child: Text(_getStatusText(isRobotConnected)),
                                ),
                              ],
                            ),
                            const SizedBox(height: 4),
                            Text(
                              _getStatusMessage(isRobotConnected),
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.8),
                                fontSize: 10,
                              ),
                            ),
                            // MQTT connection status
                            if (!isMqttConnected)
                              Padding(
                                padding: const EdgeInsets.only(top: 4),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    const Icon(
                                      Icons.cloud_off,
                                      color: Colors.orange,
                                      size: 12,
                                    ),
                                    const SizedBox(width: 4),
                                    Text(
                                      "Service Disconnected",
                                      style: TextStyle(
                                        color: Colors.orange.shade200,
                                        fontSize: 10,
                                        fontWeight: FontWeight.w500,
                                      ),
                                    ),
                                  ],
                                ),
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
                                    "Get notified of unknown faces",
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
                      // Button(
                      //   label: "Add Your Face",
                      //   press: () {
                      //     Navigator.push(
                      //       context,
                      //       MaterialPageRoute(
                      //           builder: (context) => const AddFacePage()),
                      //     );
                      //   },
                      // ),
                      
                      const SizedBox(height: 15),
                      
                      // Notifications button with count
                      InkWell(
                        onTap: _navigateToNotifications,
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(10),
                            border: mqttService.notifications.isNotEmpty 
                                ? Border.all(color: Colors.orange.withOpacity(0.5))
                                : null,
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Stack(
                                clipBehavior: Clip.none,
                                children: [
                                  const Icon(Icons.notifications_active, color: Colors.white),
                                  if (mqttService.notifications.isNotEmpty)
                                    Positioned(
                                      top: -2,
                                      right: -2,
                                      child: Container(
                                        padding: const EdgeInsets.all(2),
                                        decoration: const BoxDecoration(
                                          color: Colors.red,
                                          shape: BoxShape.circle,
                                        ),
                                        constraints: const BoxConstraints(
                                          minWidth: 14,
                                          minHeight: 14,
                                        ),
                                        child: Text(
                                          mqttService.notifications.length > 9 
                                              ? '9+' 
                                              : mqttService.notifications.length.toString(),
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 8,
                                            fontWeight: FontWeight.bold,
                                          ),
                                          textAlign: TextAlign.center,
                                        ),
                                      ),
                                    ),
                                ],
                              ),
                              const SizedBox(width: 10),
                              Text(
                                mqttService.notifications.isEmpty 
                                    ? "Check Notifications"
                                    : "View ${mqttService.notifications.length} Notification${mqttService.notifications.length == 1 ? '' : 's'}",
                                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w500),
                              ),
                            ],
                          ),
                        ),
                      ),
                      
                      const SizedBox(height: 15),
                      
                      // Connection troubleshooting button (only show if disconnected)
                      if (!isMqttConnected || !isRobotConnected)
                        InkWell(
                          onTap: () async {
                            // Show connection troubleshooting dialog
                            showDialog(
                              context: context,
                              builder: (context) => AlertDialog(
                                title: const Text('Connection Status'),
                                content: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        Icon(
                                          isMqttConnected ? Icons.check_circle : Icons.error,
                                          color: isMqttConnected ? Colors.green : Colors.red,
                                          size: 20,
                                        ),
                                        const SizedBox(width: 8),
                                        Text('Notification Service: ${isMqttConnected ? 'Connected' : 'Disconnected'}'),
                                      ],
                                    ),
                                    const SizedBox(height: 8),
                                    Row(
                                      children: [
                                        Icon(
                                          isRobotConnected ? Icons.check_circle : Icons.error,
                                          color: isRobotConnected ? Colors.green : Colors.red,
                                          size: 20,
                                        ),
                                        const SizedBox(width: 8),
                                        Text('Robot Status: ${isRobotConnected ? 'Online' : 'Offline'}'),
                                      ],
                                    ),
                                    const SizedBox(height: 16),
                                    const Text(
                                      'Troubleshooting:',
                                      style: TextStyle(fontWeight: FontWeight.bold),
                                    ),
                                    const SizedBox(height: 8),
                                    const Text('• Check your internet connection'),
                                    const Text('• Ensure your robot is powered on'),
                                    const Text('• Verify robot is connected to network'),
                                    const Text('• Try reconnecting below'),
                                  ],
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.pop(context),
                                    child: const Text('CLOSE'),
                                  ),
                                  ElevatedButton(
                                    onPressed: () async {
                                      Navigator.pop(context);
                                      await mqttService.initialize();
                                      if (mounted) {
                                        ScaffoldMessenger.of(context).showSnackBar(
                                          SnackBar(
                                            content: Text(
                                              mqttService.isConnected 
                                                  ? 'Reconnected successfully!'
                                                  : 'Reconnection failed. Please try again.',
                                            ),
                                            backgroundColor: mqttService.isConnected 
                                                ? Colors.green 
                                                : Colors.red,
                                          ),
                                        );
                                      }
                                    },
                                    child: const Text('RECONNECT'),
                                  ),
                                ],
                              ),
                            );
                          },
                          child: Container(
                            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
                            decoration: BoxDecoration(
                              color: Colors.orange.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: Colors.orange.withOpacity(0.5)),
                            ),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.help_outline, color: Colors.orange, size: 18),
                                SizedBox(width: 8),
                                Text(
                                  "Connection Issues?",
                                  style: TextStyle(color: Colors.orange, fontWeight: FontWeight.w500),
                                ),
                              ],
                            ),
                          ),
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
            );
          },
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
              type: BottomNavigationBarType.fixed,
              items: const [
                BottomNavigationBarItem(
                  icon: Icon(Icons.home), 
                  label: 'Home',
                ),
                // BottomNavigationBarItem(
                //   icon: Icon(Icons.face), 
                //   label: 'Faces',
                // ),
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