import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';
import 'dart:typed_data';
// import 'package:intl/intl.dart';
import '../JSON/users.dart';

class DatabaseHelper {
  final databaseName = "auth.db";
  
  String userTable = '''
  CREATE TABLE users (
  usrId INTEGER PRIMARY KEY AUTOINCREMENT,
  fullName TEXT,
  email TEXT UNIQUE,
  usrName TEXT UNIQUE,
  usrPassword TEXT
  )
  ''';
  
  String facesTable = '''
  CREATE TABLE faces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  image BLOB NOT NULL,
  date_added TEXT NOT NULL,
  status TEXT NOT NULL,
  user_id INTEGER,
  FOREIGN KEY (user_id) REFERENCES users(usrId)
  )
  ''';
  
  Future<Database> initDB() async {
    final databasePath = await getDatabasesPath();
    final path = join(databasePath, databaseName);
    
    return openDatabase(
      path, 
      version: 1, 
      onCreate: (db, version) async {
        await db.execute(userTable);
        await db.execute(facesTable);
      }
    );
  }
  
  Future<bool> authenticate(Users usr) async {
    final Database db = await initDB();
    var result = await db.rawQuery(
      "SELECT * FROM users WHERE usrName = ? AND usrPassword = ?",
      [usr.usrName, usr.password],
    );
    return result.isNotEmpty;
  }
  
  Future<int> createUser(Users usr) async {
    final Database db = await initDB();
    try {
      return await db.insert("users", usr.toMap());
    } catch (e) {
      return -1;
    }
  }
  
  Future<bool> checkEmailExists(String email) async {
    final Database db = await initDB();
    var result = await db.query(
      "users",
      where: "email = ?",
      whereArgs: [email],
    );
    return result.isNotEmpty;
  }
  
  Future<Users?> getUser(String usrName) async {
    final Database db = await initDB();
    var res = await db.query(
      "users",
      where: "usrName = ?",
      whereArgs: [usrName],
    );
    return res.isNotEmpty ? Users.fromMap(res.first) : null;
  }
  
  // Check if password is correct
  Future<bool> checkPassword(String username, String password) async {
    final Database db = await initDB();
    
    var res = await db.query(
      "users",
      where: "usrName = ? AND usrPassword = ?",
      whereArgs: [username, password],
    );
    
    return res.isNotEmpty;
  }
  
  // Update user information
  Future<int> updateUser(Users user) async {
    final Database db = await initDB();
    
    try {
      // Update the user record
      return await db.update(
        "users",
        user.toMap(),
        where: "usrId = ?",
        whereArgs: [user.usrId],
      );
    } catch (e) {
      // Handle unique constraint errors (email or username already exists)
      print("Error updating user: $e");
      return -1;
    }
  }
  
  // Check if username already exists (excluding current user)
  Future<bool> checkUsernameExists(String username, int? currentUserId) async {
    final Database db = await initDB();
    var result = await db.query(
      "users",
      where: "usrName = ? AND usrId != ?",
      whereArgs: [username, currentUserId ?? -1],
    );
    return result.isNotEmpty;
  }

  // FACE MANAGEMENT METHODS
  
  // Get all faces - Simplified to only fetch existing columns
  Future<List<Map<String, dynamic>>> getFaces(int? userId) async {
    final Database db = await initDB();
    
    try {
      // Simple query that only fetches columns we know exist
      return await db.rawQuery("SELECT id, name, image FROM faces");
    } catch (e) {
      print("Error fetching faces: $e");
      return []; // Return empty list if the query fails
    }
  }
  
  // Add a new face - Simplified to only include existing columns
  Future<int> addFace(String name, Uint8List imageData, int? userId) async {
    final Database db = await initDB();
    
    // Create the map of data with only essential fields
    final Map<String, dynamic> faceData = {
      'name': name,
      'image': imageData,
    };
    
    // Insert into the database
    try {
      return await db.insert('faces', faceData);
    } catch (e) {
      print("Error adding face: $e");
      return -1;
    }
  }
  
  // Delete a face - Simplified with better error handling
  Future<int> deleteFace(int faceId) async {
    final Database db = await initDB();
    
    try {
      return await db.delete(
        'faces',
        where: 'id = ?',
        whereArgs: [faceId],
      );
    } catch (e) {
      print("Error deleting face: $e");
      return 0;
    }
  }

  // Get current user (most recently logged in)
  Future<Users?> getCurrentUser() async {
    final Database db = await initDB();
    try {
      // Get the most recently logged in user
      var res = await db.query(
        "users",
        orderBy: "usrId DESC",
        limit: 1
      );
      return res.isNotEmpty ? Users.fromMap(res.first) : null;
    } catch (e) {
      print("Error getting current user: $e");
      return null;
    }
  }
}