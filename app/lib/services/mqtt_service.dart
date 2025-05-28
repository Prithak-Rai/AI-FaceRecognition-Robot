import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'package:app/Views/notification.dart';
import 'package:app/SQLite/database_helper.dart';

class MQTTService extends ChangeNotifier {
  // Debug flag for extra logging
  final bool _debug = true;
  
  // MQTT Configuration
  final String _broker = 'broker.emqx.io';
  final int _port = 1883;
  final String _clientId = 'facebot_flutter_${DateTime.now().millisecondsSinceEpoch}';
  
  // MQTT Client
  late final MqttServerClient _client;
  
  // MQTT Topics
  final String _notificationsTopic = 'facebot/notifications';
  final String _unknownFaceTopic = 'facebot/unknown_face';
  final String _statusTopic = 'facebot/status';
  final String _currentUserTopic = 'facebot/current_user';
  final String _requestCurrentUserTopic = 'facebot/request_current_user';
  
  // Connection status
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  
  // Robot status
  bool _isRobotConnected = false;
  bool get isRobotConnected => _isRobotConnected;
  
  // Connection retry count
  int _retryCount = 0;
  static const int maxRetries = 5;
  
  // Status update debouncing
  Timer? _statusDebounceTimer;
  static const Duration statusDebounceTime = Duration(seconds: 2);
  DateTime? _lastStatusUpdate;
  
  // Notifications storage
  List<UnknownFaceNotification> _notifications = [];
  List<UnknownFaceNotification> get notifications => List.unmodifiable(_notifications);
  
  final List<Function> _listeners = [];
  
  // Singleton instance
  static final MQTTService _instance = MQTTService._internal();
  
  factory MQTTService() {
    return _instance;
  }
  
  MQTTService._internal() {
    _client = MqttServerClient.withPort(
      _broker, 
      _clientId, 
      _port,
      maxConnectionAttempts: 3
    );
    
    // Initialize connection
    initialize();
  }
  
  // Initialize and connect
  Future<void> initialize() async {
    if (_isConnected) return;
    
    try {
      await _connect();
      
      // Subscribe to topics after successful connection
      _client.subscribe(_notificationsTopic, MqttQos.atLeastOnce);
      _client.subscribe(_unknownFaceTopic, MqttQos.atLeastOnce);
      _client.subscribe(_statusTopic, MqttQos.atLeastOnce);
      _client.subscribe(_currentUserTopic, MqttQos.atLeastOnce);
      _client.subscribe(_requestCurrentUserTopic, MqttQos.atLeastOnce);
      
      // Set up callback for all messages
      _client.updates?.listen((List<MqttReceivedMessage<MqttMessage>> messages) {
        for (var message in messages) {
          _handleMessage(message);
        }
      });
    } catch (e) {
      print('Error connecting to MQTT: $e');
      _isConnected = false;
    }
  }
  
  void _handleMessage(MqttReceivedMessage<MqttMessage> message) {
    try {
      final payload = MqttPublishPayload.bytesToStringAsString(
        (message.payload as MqttPublishMessage).payload.message
      );
      
      if (_debug) {
        print('Received message on topic: ${message.topic}');
        print('Message payload: $payload');
      }
      
      if (message.topic == _requestCurrentUserTopic) {
        _handleCurrentUserRequest();
      }
      // Add other message handling here
    } catch (e) {
      print('Error handling message: $e');
    }
  }
  
  Future<void> _handleCurrentUserRequest() async {
    try {
      final db = DatabaseHelper();
      final currentUser = await db.getCurrentUser();
      
      if (currentUser != null) {
        final builder = MqttClientPayloadBuilder();
        builder.addString(jsonEncode({
          'type': 'current_user_email',
          'email': currentUser.email,
          'timestamp': DateTime.now().toIso8601String()
        }));
        
        _client.publishMessage(
          _currentUserTopic,
          MqttQos.atLeastOnce,
          builder.payload!
        );
        
        if (_debug) {
          print('Sent current user email: ${currentUser.email}');
        }
      }
    } catch (e) {
      print('Error handling current user request: $e');
    }
  }
  
  Future<void> _connect() async {
    print('Connecting to MQTT broker...');
    _client.logging(on: _debug);
    _client.keepAlivePeriod = 60;
    _client.onConnected = _onConnected;
    _client.onDisconnected = _onDisconnected;
    _client.onSubscribed = _onSubscribed;
    
    _client.setProtocolV311();
    _client.autoReconnect = true;
    _client.resubscribeOnAutoReconnect = true;
    
    try {
      print('Attempting to connect to MQTT broker...');
      await _client.connect();
      
      // Wait for connection to be established
      int attempts = 0;
      while (_client.connectionStatus?.state != MqttConnectionState.connected && attempts < 10) {
        await Future.delayed(Duration(milliseconds: 100));
        attempts++;
      }
      
      if (_client.connectionStatus?.state == MqttConnectionState.connected) {
        print('Successfully connected to MQTT broker');
        _isConnected = true;
        _retryCount = 0; // Reset retry count on successful connection
        notifyListeners();
      } else {
        print('Failed to connect to MQTT broker after $attempts attempts');
        throw Exception('Failed to connect to MQTT broker');
      }
    } catch (e) {
      print('Exception connecting to MQTT broker: $e');
      _client.disconnect();
      _isConnected = false;
      _isRobotConnected = false; // Reset robot status on connection failure
      notifyListeners();
      rethrow;
    }
  }
  
  void _onConnected() {
    print('\n=== MQTT Client Connected ===');
    _isConnected = true;
    _retryCount = 0; // Reset retry count on successful connection
    notifyListeners();
    
    // Set up callback for incoming messages
    _client.updates?.listen((List<MqttReceivedMessage<MqttMessage>> messages) {
      print('\n=== Received MQTT Message ===');
      for (var message in messages) {
        final MqttPublishMessage recMess = message.payload as MqttPublishMessage;
        final String messageString = MqttPublishPayload.bytesToStringAsString(
          recMess.payload.message
        );
        
        print('Topic: ${message.topic}');
        print('QoS: ${recMess.header?.qos}');
        print('Message length: ${messageString.length} bytes');
        
        if (_debug) {
          if (messageString.length > 200) {
            print('Message preview: ${messageString.substring(0, 200)}...');
          } else {
            print('Message: $messageString');
          }
        }
        
        // Process the message
        _processMessage(message.topic, messageString);
      }
      print('=== End of Message ===\n');
    }, onError: (error) {
      print('Error in MQTT message stream: $error');
      // Handle stream error by attempting reconnection
      _onDisconnected();
    });
  }
  
  void _onDisconnected() {
    print('MQTT client disconnected');
    _isConnected = false;
    _isRobotConnected = false; // Always set robot status to offline on disconnection
    notifyListeners();
    
    // Try to reconnect with exponential backoff
    if (_retryCount < maxRetries) {
      _retryCount++;
      final delay = Duration(seconds: math.min(30, math.pow(2, _retryCount).toInt()));
      print('Attempting to reconnect in ${delay.inSeconds} seconds (attempt $_retryCount of $maxRetries)...');
      Future.delayed(delay, () {
        if (!_isConnected) {
          _connect();
        }
      });
    } else {
      print('Max retry attempts reached. Please check your connection and try again.');
    }
  }
  
  void _onSubscribed(String topic) {
    print('Successfully subscribed to topic: $topic');
  }
  
  void _processMessage(String topic, String message) {
    print('\n=== Processing MQTT Message ===');
    print('Topic: $topic');
    print('Message length: ${message.length} bytes');
    
    try {
      final Map<String, dynamic> jsonData = jsonDecode(message);
      
      if (topic == _statusTopic) {
        print('Processing status update');
        print('Status data: $jsonData');
        if (jsonData['type'] == 'robot_status') {
          final bool newStatus = jsonData['status'] == 'online';
          final DateTime timestamp = DateTime.parse(jsonData['timestamp']);
          
          // Check if we should process this status update
          if (_lastStatusUpdate != null && 
              timestamp.difference(_lastStatusUpdate!) < statusDebounceTime) {
            print('Skipping status update - too soon after last update');
            return;
          }
          
          if (_isRobotConnected != newStatus) {
            print('Robot status changed from $_isRobotConnected to $newStatus');
            
            // Cancel any existing debounce timer
            _statusDebounceTimer?.cancel();
            
            // Set up new debounce timer
            _statusDebounceTimer = Timer(statusDebounceTime, () {
              _isRobotConnected = newStatus;
              _lastStatusUpdate = DateTime.now();
              // Ensure UI updates on the main thread
              Future.microtask(() {
                print('Notifying listeners of robot status change');
                notifyListeners();
              });
            });
          } else {
            // Even if status hasn't changed, ensure UI is updated
            notifyListeners();
          }
        }
      } else if (topic == _unknownFaceTopic || topic == _notificationsTopic) {
        print('Processing notification message');
        print('Message keys: ${jsonData.keys.toList()}');
        
        // Handle both regular notifications and unknown face notifications
        if (jsonData['type'] == 'unknown_face') {
          print('Processing unknown face notification');
          print('Has image data: ${jsonData.containsKey('image') && jsonData['image'] != null}');
          
          final notification = UnknownFaceNotification.fromJson(jsonData);
          print('Notification parsed successfully:');
          print('- Type: ${notification.type}');
          print('- Title: ${notification.title}');
          print('- Body: ${notification.body}');
          print('- Timestamp: ${notification.timestamp}');
          print('- Image size: ${notification.imageBytes?.length ?? 0} bytes');
          
          // Add notification to the list
          print('Adding notification to list');
          print('Current notifications count: ${_notifications.length}');
          _notifications.insert(0, notification);
          print('New notifications count: ${_notifications.length}');
          
          // Use Future.microtask to ensure state updates happen in the next frame
          Future.microtask(() {
            print('Notifying listeners of new notification');
            notifyListeners();
            
            // Notify all custom listeners
            for (var listener in _listeners) {
              try {
                listener();
              } catch (e) {
                print('Error calling listener: $e');
              }
            }
            
            // Show system notification
            print('Showing system notification');
            _showSystemNotification(notification);
          });
        } else if (jsonData['type'] == 'alert' || jsonData['type'] == 'status' || jsonData['type'] == 'hardware_status' || jsonData['type'] == 'system_status') {
          // Handle other notification types without images
          print('Processing general notification: ${jsonData['type']}');
          
          // Check for shutdown/offline notifications and update robot status
          if (jsonData['type'] == 'system_status') {
            final String title = jsonData['title'] as String? ?? '';
            final String body = jsonData['body'] as String? ?? '';
            
            // Check if this is a shutdown notification
            if (title.toLowerCase().contains('shutting down') || 
                title.toLowerCase().contains('shutdown') ||
                body.toLowerCase().contains('shutting down') ||
                body.toLowerCase().contains('system is shutting down')) {
              print('Detected shutdown notification - setting robot to offline');
              _isRobotConnected = false;
              _lastStatusUpdate = DateTime.now();
              
              // Immediately notify listeners of status change
              Future.microtask(() {
                notifyListeners();
              });
            }
          }
          
          final notification = UnknownFaceNotification(
            type: jsonData['type'] as String,
            title: jsonData['title'] as String,
            body: jsonData['body'] as String,
            timestamp: jsonData['timestamp'] as String,
          );
          
          _notifications.insert(0, notification);
          
          Future.microtask(() {
            notifyListeners();
            for (var listener in _listeners) {
              try {
                listener();
              } catch (e) {
                print('Error calling listener: $e');
              }
            }
          });
        } else {
          print('Message type not recognized: ${jsonData['type']}');
        }
      } else {
        print('Unknown topic: $topic');
      }
    } catch (e, stackTrace) {
      print('Error processing message: $e');
      print('Stack trace: $stackTrace');
      print('Message content: $message');
      
      // Try to handle malformed JSON gracefully
      if (message.isNotEmpty) {
        print('Attempting to create fallback notification...');
        try {
          final fallbackNotification = UnknownFaceNotification(
            type: 'error',
            title: 'Message Processing Error',
            body: 'Received a message that could not be processed properly.',
            timestamp: DateTime.now().toIso8601String(),
          );
          
          _notifications.insert(0, fallbackNotification);
          Future.microtask(() {
            notifyListeners();
          });
        } catch (fallbackError) {
          print('Even fallback notification failed: $fallbackError');
        }
      }
    }
    print('=== End of Processing ===\n');
  }
  
  void _showSystemNotification(UnknownFaceNotification notification) {
    // This method will be implemented to show system notifications
    // when the app is in the background
    print('System notification would be shown here');
    print('- Title: ${notification.title}');
    print('- Body: ${notification.body}');
    print('- Has image: ${notification.imageBytes != null}');
  }
  
  void clearNotifications() {
    _notifications.clear();
    // Use Future.microtask for UI updates
    Future.microtask(() {
      notifyListeners();
    });
  }
  
  // Helper method to manually test MQTT connection by sending a message
  void sendTestMessage() {
    if (_client.connectionStatus?.state == MqttConnectionState.connected) {
      final builder = MqttClientPayloadBuilder();
      builder.addString(json.encode({
        'type': 'test',
        'title': 'Test Message',
        'body': 'This is a test message from Flutter app',
        'timestamp': DateTime.now().toIso8601String()
      }));
      
      _client.publishMessage(_notificationsTopic, MqttQos.atLeastOnce, builder.payload!);
      print('Test message sent!');
    } else {
      print('Cannot send test message: client not connected');
    }
  }
  
  void addListener(Function listener) {
    _listeners.add(listener);
  }
  
  void removeListener(Function listener) {
    _listeners.remove(listener);
  }
  
  @override
  void dispose() {
    _statusDebounceTimer?.cancel();
    _client.disconnect();
    super.dispose();
  }
}