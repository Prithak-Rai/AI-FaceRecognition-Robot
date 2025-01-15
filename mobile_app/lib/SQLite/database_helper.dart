import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';
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

  Future<Database> initDB() async {
    final databasePath = await getDatabasesPath();
    final path = join(databasePath, databaseName);

    return openDatabase(path, version: 1, onCreate: (db, version) async {
      await db.execute(userTable);
    });
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
}
