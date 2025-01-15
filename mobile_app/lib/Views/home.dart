import 'package:flutter/material.dart';
import 'package:flutter_sqlite_auth_app/Views/add_face.dart';
import 'package:flutter_sqlite_auth_app/Components/colors.dart';
import 'package:flutter_sqlite_auth_app/Components/button.dart';
import 'package:flutter_sqlite_auth_app/Views/profile.dart';
import 'package:flutter_sqlite_auth_app/JSON/users.dart';

class HomePage extends StatelessWidget {
  final Users? profile;  // Accepting the profile data here

  const HomePage({Key? key, this.profile}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: primaryColor,
        leading: GestureDetector(
          onTap: () {
            // Navigate to the profile page
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => Profile(profile: profile)),
            );
          },
          child: const Padding(
            padding: EdgeInsets.all(8.0),
            child: CircleAvatar(
              backgroundColor: Colors.white,
              radius: 18,
              backgroundImage: AssetImage("assets/no_user.jpg"),
            ),
          ),
        ),
        title: const Text(
          'Home',
          style: TextStyle(fontSize: 22, color: Colors.white),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.menu, size: 30, color: Colors.white),
            onPressed: () {
              
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 45.0, horizontal: 20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 10),
                const Text(
                  'Welcome to the App!',
                  style: TextStyle(fontSize: 28, color: primaryColor),
                ),
                const SizedBox(height: 20),
                Button(
                  label: "Add Face",
                  press: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const AddFacePage()),
                    );
                  },
                ),
                const SizedBox(height: 20),
                const ListTile(
                  leading: Icon(Icons.add, size: 30, color: primaryColor),
                  title: Text(
                    "Add a Face to the System",
                    style: TextStyle(fontSize: 18),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
