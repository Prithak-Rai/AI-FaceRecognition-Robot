import 'package:flutter/material.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  _SettingsPageState createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  // Default values for sliders
  double _movementSpeed = 50;
  double _recognitionSensitivity = 70;
  bool _enableNotifications = true;
  bool _enableSounds = true;

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
          title: const Text('Settings', style: TextStyle(color: Colors.white)),
          iconTheme: const IconThemeData(color: Colors.white),
        ),
        body: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Robot Movement Section
                _buildSectionHeader('Robot Movement'),
                Card(
                  color: Colors.white.withOpacity(0.2),
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Movement Speed',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 5),
                        const Text(
                          'Adjust how fast the servos move',
                          style: TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                        Slider(
                          value: _movementSpeed,
                          min: 0,
                          max: 100,
                          divisions: 10,
                          activeColor: Colors.white,
                          inactiveColor: Colors.white30,
                          label: _movementSpeed.round().toString(),
                          onChanged: (value) {
                            setState(() {
                              _movementSpeed = value;
                            });
                          },
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: const [
                            Text('Slow', style: TextStyle(color: Colors.white70, fontSize: 12)),
                            Text('Fast', style: TextStyle(color: Colors.white70, fontSize: 12)),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                
                const SizedBox(height: 20),
                
                // Recognition Settings
                _buildSectionHeader('Facial Recognition'),
                Card(
                  color: Colors.white.withOpacity(0.2),
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Recognition Sensitivity',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 5),
                        const Text(
                          'Adjust facial recognition confidence threshold',
                          style: TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                        Slider(
                          value: _recognitionSensitivity,
                          min: 0,
                          max: 100,
                          divisions: 10,
                          activeColor: Colors.white,
                          inactiveColor: Colors.white30,
                          label: _recognitionSensitivity.round().toString(),
                          onChanged: (value) {
                            setState(() {
                              _recognitionSensitivity = value;
                            });
                          },
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: const [
                            Text('Low', style: TextStyle(color: Colors.white70, fontSize: 12)),
                            Text('High', style: TextStyle(color: Colors.white70, fontSize: 12)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          icon: const Icon(Icons.delete_outline),
                          label: const Text('Remove Face Data'),
                          style: ElevatedButton.styleFrom(
                            foregroundColor: Colors.white,
                            backgroundColor: Colors.red.withOpacity(0.6),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                          ),
                          onPressed: () {
                            _showDeleteConfirmationDialog();
                          },
                        ),
                      ],
                    ),
                  ),
                ),
                
                const SizedBox(height: 20),
                
                // Raspberry Pi Settings
                _buildSectionHeader('Device Settings'),
                Card(
                  color: Colors.white.withOpacity(0.2),
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        ListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text(
                            'Enable Notifications',
                            style: TextStyle(color: Colors.white),
                          ),
                          trailing: Switch(
                            value: _enableNotifications,
                            onChanged: (value) {
                              setState(() {
                                _enableNotifications = value;
                              });
                            },
                            activeColor: Colors.white,
                            activeTrackColor: Colors.white38,
                          ),
                        ),
                        ListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text(
                            'Enable Sounds',
                            style: TextStyle(color: Colors.white),
                          ),
                          trailing: Switch(
                            value: _enableSounds,
                            onChanged: (value) {
                              setState(() {
                                _enableSounds = value;
                              });
                            },
                            activeColor: Colors.white,
                            activeTrackColor: Colors.white38,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Center(
                          child: ElevatedButton.icon(
                            icon: const Icon(Icons.restart_alt),
                            label: const Text('Restart Raspberry Pi'),
                            style: ElevatedButton.styleFrom(
                              foregroundColor: Colors.white,
                              backgroundColor: Colors.blue.withOpacity(0.6),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                            onPressed: () {
                              _showRestartConfirmationDialog();
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                
                const SizedBox(height: 20),
                
                // About section
                _buildSectionHeader('About'),
                Card(
                  color: Colors.white.withOpacity(0.2),
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const ListTile(
                          contentPadding: EdgeInsets.zero,
                          title: Text('Version', style: TextStyle(color: Colors.white)),
                          trailing: Text('1.0.0', style: TextStyle(color: Colors.white70)),
                        ),
                        const ListTile(
                          contentPadding: EdgeInsets.zero,
                          title: Text('Device IP', style: TextStyle(color: Colors.white)),
                          trailing: Text('192.168.1.100', style: TextStyle(color: Colors.white70)),
                        ),
                        Center(
                          child: TextButton(
                            onPressed: () {
                              // Show about dialog or navigate to about page
                            },
                            child: const Text(
                              'Privacy Policy',
                              style: TextStyle(color: Colors.white),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(left: 16.0, bottom: 8.0),
      child: Text(
        title,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  void _showDeleteConfirmationDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.white.withOpacity(0.9),
        title: const Text('Remove Face Data'),
        content: const Text(
          'This will delete all stored facial recognition data. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              // Implement face data deletion logic here
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Face data deleted')),
              );
            },
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _showRestartConfirmationDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.white.withOpacity(0.9),
        title: const Text('Restart Raspberry Pi'),
        content: const Text(
          'Are you sure you want to restart the Raspberry Pi? This will temporarily disconnect the robot.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              // Implement restart logic here
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Restarting Raspberry Pi...')),
              );
            },
            child: const Text('Restart', style: TextStyle(color: Colors.blue)),
          ),
        ],
      ),
    );
  }
}