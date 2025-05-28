import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'package:app/Components/button.dart';
import 'package:app/JSON/users.dart';
import 'package:app/Views/add_face.dart';
import 'package:app/Views/home.dart';
import 'package:app/Views/profile.dart';
import 'package:app/Views/settings.dart';
import 'package:app/SQLite/database_helper.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'dart:convert';
import 'package:provider/provider.dart';
import 'package:app/services/mqtt_service.dart';

class FacesScreen extends StatefulWidget {
  const FacesScreen({Key? key}) : super(key: key);

  @override
  State<FacesScreen> createState() => _FacesScreenState();
}

class _FacesScreenState extends State<FacesScreen> {
  @override
  void initState() {
    super.initState();
    // Initialize MQTT connection if not already connected
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final mqttService = Provider.of<MQTTService>(context, listen: false);
      if (!mqttService.isConnected) {
        mqttService.initialize();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Known Faces'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Face list refresh not available')),
              );
            },
          ),
        ],
      ),
      body: Consumer<MQTTService>(
        builder: (context, mqttService, child) {
          if (!mqttService.isConnected) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.cloud_off, size: 64, color: Colors.grey),
                  const SizedBox(height: 16),
                  const Text(
                    'Not connected to MQTT broker',
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Check your connection and restart the app',
                    style: TextStyle(fontSize: 14, color: Colors.grey),
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton.icon(
                    icon: const Icon(Icons.refresh),
                    label: const Text('Reconnect'),
                    onPressed: () async {
                      await mqttService.initialize();
                    },
                  ),
                ],
              ),
            );
          }

          return const Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.face, size: 64, color: Colors.grey),
                SizedBox(height: 16),
                Text(
                  'Face Management',
                  style: TextStyle(fontSize: 16, color: Colors.grey),
                ),
                SizedBox(height: 8),
                Text(
                  'Face management features are currently disabled',
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}