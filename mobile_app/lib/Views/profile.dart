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
      body: BottomBar(
        body: (context, controller) => SingleChildScrollView(
          controller: controller,
          child: Column(
            children: [
              // Gradient Header with Back Button
              Container(
                height: 180,
                width: double.infinity,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topRight,
                    end: Alignment.bottomLeft,
                    colors: [
                      Colors.blue,
                      Color(0xFFFFA2A2),
                    ],
                  ),
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(30),
                    bottomRight: Radius.circular(30),
                  ),
                ),
                child: SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.arrow_back, color: Colors.white),
                          onPressed: () => Navigator.pop(context),
                        ),
                        const Expanded(
                          child: Text(
                            "User Profile",
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.white
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.edit, color: Colors.white),
                          onPressed: () {
                            // Navigate to edit profile
                          },
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              
              // Profile Picture
              Transform.translate(
                offset: const Offset(0, -50),
                child: Container(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 5),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.3),
                        spreadRadius: 2,
                        blurRadius: 8,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: const CircleAvatar(
                    backgroundColor: Color.fromARGB(255, 106, 105, 105),
                    radius: 50,
                    child: CircleAvatar(
                      backgroundImage: AssetImage("assets/no_user.jpg"),
                      radius: 48,
                    ),
                  ),
                ),
              ),
              
              // User Name and Email
              Transform.translate(
                offset: const Offset(0, -40),
                child: Column(
                  children: [
                    Text(
                      profile?.fullName ?? "Guest User",
                      style: const TextStyle(
                        fontSize: 24, 
                        fontWeight: FontWeight.bold,
                        color: Colors.black
                      ),
                    ),
                    const SizedBox(height: 5),
                    Text(
                      profile?.email ?? "No Email",
                      style: const TextStyle(
                        fontSize: 16, 
                        color: Colors.black54
                      ),
                    ),
                  ],
                ),
              ),
              
              // Divider
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Divider(
                  color: Colors.grey.withOpacity(0.3),
                  thickness: 1,
                ),
              ),
              
              // Profile Info - Only DB Info
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 10),
                      child: Text(
                        "Profile Information",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    
                    _buildProfileInfo(Icons.person, "Full Name", profile?.fullName ?? "N/A"),
                    _buildProfileInfo(Icons.email, "Email", profile?.email ?? "N/A"),
                    _buildProfileInfo(Icons.account_circle, "Username", profile?.usrName ?? "N/A"),
                  ],
                ),
              ),
                            
              const SizedBox(height: 20),
              
              // Sign Out Button
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Button(
                  label: "SIGN OUT",
                  press: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const LoginScreen()),
                    );
                  },
                ),
              ),
              
              // Extra space at bottom to avoid overlap with bottom bar
              const SizedBox(height: 100),
            ],
          ),
        ),
        borderRadius: BorderRadius.circular(25),
        width: MediaQuery.of(context).size.width * 0.8,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
        showIcon: true,
        iconHeight: 35,
        iconWidth: 300,
        end: 20,
        alignment: Alignment.bottomCenter,
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(25),
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.3),
                spreadRadius: 2,
                blurRadius: 10,
                offset: const Offset(0, -3),
              ),
            ],
          ),
          child: BottomNavigationBar(
            backgroundColor: Colors.transparent,
            elevation: 0,
            selectedItemColor: Colors.blue,
            unselectedItemColor: Colors.grey,
            currentIndex: 1,
            onTap: (index) {
              if (index == 0) {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => HomePage(profile: profile)),
                );
              } else if (index == 2) {
                // Navigation to settings would go here
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
      ),
    );
  }

  Widget _buildProfileInfo(IconData icon, String subtitle, String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: ListTile(
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.blue.withOpacity(0.1),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(icon, size: 24, color: Colors.blue),
        ),
        subtitle: Text(
          subtitle, 
          style: const TextStyle(color: Colors.black54, fontSize: 12)
        ),
        title: Text(
          title, 
          style: const TextStyle(fontSize: 16, color: Colors.black, fontWeight: FontWeight.w500)
        ),
        contentPadding: EdgeInsets.zero,
        dense: true,
        visualDensity: const VisualDensity(vertical: -1),
      ),
    );
  }
}