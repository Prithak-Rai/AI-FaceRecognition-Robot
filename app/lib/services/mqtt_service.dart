import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

// Model class for notifications
class UnknownFaceNotification {
  final String type;
  final String title;
  final String body;
  final String timestamp;
  final String? imageBase64;

  UnknownFaceNotification({
    required this.type,
    required this.title,
    required this.body,
    required this.timestamp,
    this.imageBase64,
  });

  factory UnknownFaceNotification.fromJson(Map<String, dynamic> json) {
    return UnknownFaceNotification(
      type: json['type'] ?? 'notification',
      title: json['title'] ?? 'Notification',
      body: json['body'] ?? 'No description available',
      timestamp: json['timestamp'] ?? DateTime.now().toIso8601String(),
      imageBase64: json['image'],  // This is the image from the Python backend
    );
  }
  
  // Helper method to decode base64 image
  Uint8List? get imageBytes {
    if (imageBase64 == null || imageBase64!.isEmpty) {
      return null;
    }
    
    try {
      // Sanitize the Base64 string - remove any potential padding indicators
      String sanitized = imageBase64!.trim();
      
      // Check if it's a proper base64 string
      if (sanitized.length % 4 > 0) {
        // Add padding if needed
        sanitized = sanitized.padRight(
          sanitized.length + (4 - sanitized.length % 4),
          '='
        );
      }
      
      // Check if the string has data: prefixes (common in web)
      if (sanitized.contains(',')) {
        sanitized = sanitized.split(',').last;
      }
      
      return base64Decode(sanitized);
    } catch (e) {
      print('Error decoding image: $e');
      return null;
    }
  }
}

class MQTTService extends ChangeNotifier {
  // Debug flag for extra logging
  final bool _debug = true;
  // MQTT Client
  MqttServerClient? _client;
  final String _broker = 'broker.emqx.io';
  final int _port = 1883;
  final String _clientId = 'facebot_flutter_${DateTime.now().millisecondsSinceEpoch}';
  
  // MQTT Topics
  final String _notificationsTopic = 'facebot/notifications';
  final String _unknownFaceTopic = 'facebot/unknown_face';
  
  // Connection status
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  
  // Notifications storage
  List<UnknownFaceNotification> _notifications = [];
  List<UnknownFaceNotification> get notifications => _notifications;
  
  // Initialize and connect
  Future<void> initialize() async {
    await _connect();
  }
  
  Future<void> _connect() async {
    // Create client with maxConnectionAttempts set during initialization
    _client = MqttServerClient.withPort(
      _broker, 
      _clientId, 
      _port,
      maxConnectionAttempts: 3
    );
    
    // Set connection options
    _client!.logging(on: _debug);
    _client!.keepAlivePeriod = 60;
    _client!.onConnected = _onConnected;
    _client!.onDisconnected = _onDisconnected;
    
    // Set larger message size limit for images
    _client!.setProtocolV311();
    _client!.autoReconnect = true;
    
    try {
      await _client!.connect();
    } catch (e) {
      print('Exception connecting to MQTT broker: $e');
      _client?.disconnect();
    }
  }
  
  void _onConnected() {
    print('MQTT client connected');
    _isConnected = true;
    notifyListeners();
    
    // Subscribe to topics
    _client!.subscribe(_notificationsTopic, MqttQos.atLeastOnce);
    print('Subscription confirmed for topic $_notificationsTopic');
    
    _client!.subscribe(_unknownFaceTopic, MqttQos.atLeastOnce);
    print('Subscription confirmed for topic $_unknownFaceTopic');
    
    // Set up callback for incoming messages
    _client!.updates!.listen((List<MqttReceivedMessage<MqttMessage>> messages) {
      for (var message in messages) {
        final MqttPublishMessage recMess = message.payload as MqttPublishMessage;
        
        // Use utf8 decoder for the message payload
        final String messageString = MqttPublishPayload.bytesToStringAsString(
          recMess.payload.message
        );
        
        print('Received message from topic: ${message.topic}');
        if (_debug) {
          print('Message length: ${messageString.length} bytes');
          if (messageString.length > 200) {
            print('Message preview: ${messageString.substring(0, 200)}...');
          } else {
            print('Message: $messageString');
          }
        }
        
        // Process the message
        _processMessage(message.topic, messageString);
      }
    });
  }
  
  void _onDisconnected() {
    print('MQTT client disconnected');
    _isConnected = false;
    notifyListeners();
    
    // Try to reconnect after a delay
    Future.delayed(Duration(seconds: 5), () {
      print('Attempting to reconnect...');
      _connect();
    });
  }
  
  void _processMessage(String topic, String message) {
    try {
      // Parse the JSON message
      final Map<String, dynamic> jsonData = jsonDecode(message);
      
      if (_debug) {
        print('Topic: $topic');
        print('Message keys: ${jsonData.keys.toList()}');
      }
      
      // Create notification object
      final notification = UnknownFaceNotification.fromJson(jsonData);
      
      // Verify image data for debugging
      if (notification.imageBase64 != null && notification.imageBase64!.isNotEmpty) {
        print('Image data found in notification: ${notification.imageBase64!.length} bytes');
        
        // Test decoding the image
        final testImage = notification.imageBytes;
        if (testImage != null) {
          print('✅ Successfully decoded image: ${testImage.length} bytes');
        } else {
          print('❌ Failed to decode image from notification');
        }
      } else {
        print('No image data in notification');
      }
      
      // Add to notifications list
      _notifications.insert(0, notification);  // Add at the beginning (newest first)
      
      // Update UI
      notifyListeners();
      
      print('Added new notification: ${notification.title}');
    } catch (e) {
      print('Error processing message: $e');
    }
  }
  
  void clearNotifications() {
    _notifications.clear();
    notifyListeners();
  }
  
  // Helper method to manually test MQTT connection by sending a message
  void sendTestMessage() {
    if (_client?.connectionStatus?.state == MqttConnectionState.connected) {
      final builder = MqttClientPayloadBuilder();
      builder.addString(json.encode({
        'type': 'test',
        'title': 'Test Message',
        'body': 'This is a test message from Flutter app',
        'timestamp': DateTime.now().toIso8601String()
      }));
      
      _client?.publishMessage(_notificationsTopic, MqttQos.atLeastOnce, builder.payload!);
      print('Test message sent!');
    } else {
      print('Cannot send test message: client not connected');
    }
  }
  
  @override
  void dispose() {
    _client?.disconnect();
    super.dispose();
  }
}