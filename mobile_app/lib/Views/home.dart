import 'package:flutter/material.dart';
import 'package:flutter_sqlite_auth_app/Views/add_face.dart';
// import 'package:flutter_sqlite_auth_app/Components/colors.dart';
import 'package:flutter_sqlite_auth_app/Components/button.dart';
import 'package:flutter_sqlite_auth_app/Views/profile.dart';
import 'package:flutter_sqlite_auth_app/JSON/users.dart';

class HomePage extends StatefulWidget {
  final Users? profile;

  const HomePage({Key? key, this.profile}) : super(key: key);

  @override
 
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });

    if (index == 1) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => Profile(profile: widget.profile),
        ),
      );
    } else if (index == 2) {
      
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
              padding: const EdgeInsets.symmetric(vertical: 45.0, horizontal: 20),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 10),
                  const Text(
                    'Welcome to the App!',
                    style: TextStyle(fontSize: 28, color: Colors.white),
                  ),
                  const SizedBox(height: 20),
                  Button(
                    label: "Add Face",
                    press: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const AddFacePage()),
                      );
                    },
                  ),
                  const SizedBox(height: 20),
                  const ListTile(
                    leading: Icon(Icons.add, size: 30, color: Colors.white),
                    title: Text(
                      "Add a Face to the System",
                      style: TextStyle(fontSize: 18, color: Colors.white),
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
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topRight, 
                end: Alignment.bottomLeft,
                colors: [
                  Colors.transparent,
                  Colors.transparent, 
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
              items: const [
                BottomNavigationBarItem(
                  icon: Icon(Icons.home), 
                  label: 'Home',
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
