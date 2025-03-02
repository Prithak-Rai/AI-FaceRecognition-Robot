import 'package:flutter/material.dart';
import 'package:flutter_sqlite_auth_app/Components/button.dart';
import 'package:flutter_sqlite_auth_app/JSON/users.dart';
import 'package:flutter_sqlite_auth_app/Views/login.dart';
import 'package:flutter_sqlite_auth_app/Views/home.dart';
import 'package:flutter_floating_bottom_bar/flutter_floating_bottom_bar.dart';

class Profile extends StatelessWidget {
  final Users? profile;

  const Profile({super.key, this.profile});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: BottomBar(
        body: (context, controller) => SingleChildScrollView(
          controller: controller,
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 120.0, horizontal: 20), // Reduced vertical padding to 0
              child: Column(
                mainAxisAlignment: MainAxisAlignment.start, // Align to the top
                crossAxisAlignment: CrossAxisAlignment.center, // Ensures everything starts from the left
                children: [
                  Stack(
                    clipBehavior: Clip.none,
                    children: [
                      Positioned(
                        top: -725,
                        left: -250,
                        right: -250,
                        child: Container(
                          height: 1000,
                          width: 1000,
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            gradient: LinearGradient(
                              begin: Alignment.topRight,
                              end: Alignment.bottomLeft,
                              colors: [
                                Colors.blue,
                                Color(0xFFFFA2A2),
                              ],
                            ),
                          ),
                        ),
                      ),
                      const CircleAvatar(
                        backgroundColor: Color.fromARGB(255, 106, 105, 105),
                        radius: 77,
                        child: CircleAvatar(
                          backgroundImage: AssetImage("assets/no_user.jpg"),
                          radius: 75,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(
                    profile?.fullName ?? "Guest User",
                    style: const TextStyle(fontSize: 28, color: Colors.black),
                  ),
                  Text(
                    profile?.email ?? "No Email",
                    style: const TextStyle(fontSize: 17, color: Colors.black87),
                  ),
                  const SizedBox(height: 20),
                  Button(
                    label: "SIGN OUT",
                    press: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => const LoginScreen()),
                      );
                    },
                  ),
                  const SizedBox(height: 20),
                  _buildProfileInfo(Icons.person, "Full Name", profile?.fullName ?? "N/A"),
                  _buildProfileInfo(Icons.email, "Email", profile?.email ?? "N/A"),
                  _buildProfileInfo(Icons.account_circle, "Username", profile?.usrName ?? "admin"),
                ],
              ),
            ),
          ),
        ),
        child: BottomNavigationBar(
          backgroundColor: Colors.white,
          elevation: 10,
          selectedItemColor: Colors.blue,
          unselectedItemColor: Colors.grey,
          currentIndex: 1,
          onTap: (index) {
            if (index == 0) {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (context) => const HomePage()),
              );
            }
          },
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
    );
  }

  Widget _buildProfileInfo(IconData icon, String subtitle, String title) {
    return ListTile(
      leading: Icon(icon, size: 30, color: Colors.black),
      subtitle: Text(subtitle, style: const TextStyle(color: Colors.black54)),
      title: Text(title, style: const TextStyle(fontSize: 18, color: Colors.black)),
    );
  }
}
