import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:app/services/mqtt_service.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({Key? key}) : super(key: key);

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> with WidgetsBindingObserver {
  final GlobalKey<ScaffoldMessengerState> _scaffoldMessengerKey = GlobalKey<ScaffoldMessengerState>();
  
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    
    // Initialize MQTT connection if not already connected
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final mqttService = Provider.of<MQTTService>(context, listen: false);
      if (!mqttService.isConnected) {
        mqttService.initialize().then((_) {
          _showConnectionStatus(mqttService.isConnected);
        });
      }
      
      // Set up notification listener
      mqttService.addListener(_onNotificationReceived);
    });
  }
  
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Remove listener to avoid memory leaks
    Provider.of<MQTTService>(context, listen: false).removeListener(_onNotificationReceived);
    super.dispose();
  }
  
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // When app comes to foreground, refresh MQTT connection if needed
    if (state == AppLifecycleState.resumed) {
      final mqttService = Provider.of<MQTTService>(context, listen: false);
      if (!mqttService.isConnected) {
        mqttService.initialize();
      }
    }
  }
  
  void _onNotificationReceived() {
    // Get the latest notification (if any)
    final mqttService = Provider.of<MQTTService>(context, listen: false);
    final notifications = mqttService.notifications;
    
    if (notifications.isNotEmpty) {
      final latestNotification = notifications.first;
      
      // Show a snackbar when a new notification arrives
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('New notification: ${latestNotification.title}'),
              duration: const Duration(seconds: 3),
              action: SnackBarAction(
                label: 'VIEW',
                onPressed: () {
                  _showDetailedView(context, latestNotification);
                },
              ),
            ),
          );
        }
      });
    }
  }
  
  void _showConnectionStatus(bool isConnected) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(isConnected 
            ? 'Connected to notification service' 
            : 'Failed to connect to notification service'),
          backgroundColor: isConnected ? Colors.green : Colors.red,
          duration: const Duration(seconds: 3),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return ScaffoldMessenger(
      key: _scaffoldMessengerKey,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Notifications'),
          actions: [
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: () {
                Provider.of<MQTTService>(context, listen: false).sendTestMessage();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Test request sent')),
                );
              },
            ),
            IconButton(
              icon: const Icon(Icons.delete_sweep),
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (context) => AlertDialog(
                    title: const Text('Clear Notifications'),
                    content: const Text('Are you sure you want to clear all notifications?'),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.pop(context),
                        child: const Text('CANCEL'),
                      ),
                      TextButton(
                        onPressed: () {
                          Provider.of<MQTTService>(context, listen: false).clearNotifications();
                          Navigator.pop(context);
                        },
                        child: const Text('CLEAR'),
                      ),
                    ],
                  ),
                );
              },
            ),
          ],
        ),
        body: Consumer<MQTTService>(
          builder: (context, mqttService, child) {
            final notifications = mqttService.notifications;
            
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
                        _showConnectionStatus(mqttService.isConnected);
                      },
                    ),
                  ],
                ),
              );
            }
            
            if (notifications.isEmpty) {
              return const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.notifications_off, size: 64, color: Colors.grey),
                    SizedBox(height: 16),
                    Text(
                      'No notifications yet',
                      style: TextStyle(fontSize: 16, color: Colors.grey),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Notifications from your FaceBot will appear here',
                      style: TextStyle(fontSize: 14, color: Colors.grey),
                    ),
                  ],
                ),
              );
            }
            
            return RefreshIndicator(
              onRefresh: () async {
                await Future.delayed(const Duration(milliseconds: 500));
                if (!mqttService.isConnected) {
                  await mqttService.initialize();
                }
              },
              child: ListView.builder(
                itemCount: notifications.length,
                itemBuilder: (context, index) {
                  final notification = notifications[index];
                  
                  DateTime timestamp;
                  try {
                    timestamp = DateTime.parse(notification.timestamp);
                  } catch (e) {
                    timestamp = DateTime.now();
                  }
                  
                  final formattedDate = DateFormat('MMM d, yyyy - h:mm a').format(timestamp);
                  
                  IconData notificationIcon;
                  Color iconColor;
                  
                  switch (notification.type.toLowerCase()) {
                    case 'unknown_face':
                      notificationIcon = Icons.face;
                      iconColor = Colors.red;
                      break;
                    case 'test':
                      notificationIcon = Icons.bug_report;
                      iconColor = Colors.green;
                      break;
                    case 'status':
                      notificationIcon = Icons.info;
                      iconColor = Colors.blue;
                      break;
                    case 'hardware_status':
                      notificationIcon = Icons.device_hub;
                      iconColor = Colors.orange;
                      break;
                    default:
                      notificationIcon = Icons.notifications;
                      iconColor = Colors.purple;
                  }
                  
                  final hasValidImage = notification.imageBytes != null;
                  
                  return Hero(
                    tag: 'notification_${notification.timestamp}',
                    child: Card(
                      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      elevation: 2,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          ListTile(
                            leading: CircleAvatar(
                              backgroundColor: iconColor.withOpacity(0.2),
                              child: Icon(notificationIcon, color: iconColor),
                            ),
                            title: Text(notification.title),
                            subtitle: Text(formattedDate),
                            trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                            onTap: () {
                              _showDetailedView(context, notification);
                            },
                          ),
                          
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                            child: Text(notification.body),
                          ),
                          
                          if (hasValidImage)
                            InkWell(
                              onTap: () {
                                _showDetailedView(context, notification);
                              },
                              child: Container(
                                width: double.infinity,
                                height: 150,
                                margin: const EdgeInsets.all(16),
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(8),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.1),
                                      blurRadius: 4,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(8),
                                  child: Stack(
                                    children: [
                                      Positioned.fill(
                                        child: Builder(
                                          builder: (context) {
                                            try {
                                              return Image.memory(
                                                notification.imageBytes!,
                                                fit: BoxFit.cover,
                                                errorBuilder: (context, error, stackTrace) {
                                                  print('Error rendering image: $error');
                                                  return const Center(
                                                    child: Icon(Icons.broken_image, size: 50, color: Colors.grey),
                                                  );
                                                },
                                              );
                                            } catch (e) {
                                              print('Exception loading image: $e');
                                              return const Center(
                                                child: Icon(Icons.error_outline, size: 50, color: Colors.red),
                                              );
                                            }
                                          },
                                        ),
                                      ),
                                      Positioned(
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(vertical: 4),
                                          color: Colors.black54,
                                          child: const Center(
                                            child: Text(
                                              'Tap to view details',
                                              style: TextStyle(color: Colors.white, fontSize: 12),
                                            ),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          
                          if (notification.imageBase64 != null && notification.imageBase64!.isNotEmpty && !hasValidImage)
                            Container(
                              width: double.infinity,
                              margin: const EdgeInsets.all(16),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.red.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(color: Colors.red.withOpacity(0.3))
                              ),
                              child: Row(
                                children: const [
                                  Icon(Icons.warning_amber_rounded, color: Colors.red),
                                  SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      'Image failed to load. There may be an issue with the image format.',
                                      style: TextStyle(color: Colors.red),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            );
          },
        ),
      ),
    );
  }
  
  void _showDetailedView(BuildContext context, UnknownFaceNotification notification) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DetailedNotificationScreen(notification: notification),
      ),
    );
  }
}

class DetailedNotificationScreen extends StatelessWidget {
  final UnknownFaceNotification notification;
  
  const DetailedNotificationScreen({Key? key, required this.notification}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    DateTime timestamp;
    try {
      timestamp = DateTime.parse(notification.timestamp);
    } catch (e) {
      timestamp = DateTime.now();
    }
    
    final formattedDate = DateFormat('MMMM d, yyyy - h:mm:ss a').format(timestamp);
    final hasValidImage = notification.imageBytes != null;
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Details'),
        actions: [
          if (hasValidImage)
            IconButton(
              icon: const Icon(Icons.fullscreen),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => Scaffold(
                      appBar: AppBar(
                        backgroundColor: Colors.black,
                        foregroundColor: Colors.white,
                        title: const Text('Image View'),
                      ),
                      backgroundColor: Colors.black,
                      body: Center(
                        child: InteractiveViewer(
                          minScale: 0.5,
                          maxScale: 3.0,
                          child: Image.memory(
                            notification.imageBytes!,
                            fit: BoxFit.contain,
                          ),
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (hasValidImage)
              Hero(
                tag: 'notification_${notification.timestamp}',
                child: Container(
                  width: double.infinity,
                  height: 300,
                  decoration: const BoxDecoration(
                    color: Colors.black,
                  ),
                  child: Builder(
                    builder: (context) {
                      try {
                        return Image.memory(
                          notification.imageBytes!,
                          fit: BoxFit.contain,
                          errorBuilder: (context, error, stackTrace) {
                            print('Error rendering full image: $error');
                            return const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(Icons.broken_image, size: 64, color: Colors.grey),
                                  SizedBox(height: 16),
                                  Text('Failed to load image', style: TextStyle(color: Colors.white70)),
                                ],
                              ),
                            );
                          },
                        );
                      } catch (e) {
                        print('Exception loading full image: $e');
                        return const Center(
                          child: Icon(Icons.error_outline, size: 64, color: Colors.red),
                        );
                      }
                    },
                  ),
                ),
              ),
            
            if (notification.imageBase64 != null && notification.imageBase64!.isNotEmpty && !hasValidImage)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                color: Colors.red.withOpacity(0.1),
                child: Column(
                  children: const [
                    Icon(Icons.image_not_supported, size: 64, color: Colors.red),
                    SizedBox(height: 16),
                    Text(
                      'Image Failed to Load',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.red,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'The image data could not be decoded properly. There may be an issue with the format or transmission.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.red),
                    ),
                  ],
                ),
              ),
            
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    notification.title,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  
                  const SizedBox(height: 8),
                  
                  Row(
                    children: [
                      const Icon(Icons.access_time, size: 18, color: Colors.grey),
                      const SizedBox(width: 8),
                      Text(
                        formattedDate,
                        style: const TextStyle(
                          color: Colors.grey,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 16),
                  const Divider(),
                  const SizedBox(height: 16),
                  
                  const Text(
                    'Description',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    notification.body,
                    style: const TextStyle(fontSize: 16),
                  ),
                  
                  const SizedBox(height: 32),
                  
                  const Text(
                    'Actions',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      if (hasValidImage)
                        _buildActionButton(
                          context,
                          icon: Icons.save_alt,
                          label: 'Save Image',
                          onTap: () {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('Image saved to gallery')),
                            );
                          },
                        ),
                      _buildActionButton(
                        context,
                        icon: Icons.share,
                        label: 'Share',
                        onTap: () {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Sharing not implemented yet')),
                          );
                        },
                      ),
                      _buildActionButton(
                        context,
                        icon: Icons.delete,
                        label: 'Delete',
                        onTap: () {
                          Navigator.pop(context);
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Notification deleted')),
                          );
                        },
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildActionButton(
    BuildContext context, {
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(30),
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Icon(icon, size: 24),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(fontSize: 12),
        ),
      ],
    );
  }
}