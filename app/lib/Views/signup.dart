import 'package:flutter/material.dart';
import 'package:app/Components/button.dart';
import 'package:app/Components/colors.dart';
import 'package:app/Components/textfield.dart';
import 'package:app/JSON/users.dart';
import 'package:app/Views/login.dart';
import 'package:app/Views/home.dart'; 

import '../SQLite/database_helper.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final fullName = TextEditingController();
  final email = TextEditingController();
  final usrName = TextEditingController();
  final password = TextEditingController();
  final confirmPassword = TextEditingController();
  final db = DatabaseHelper();

  bool isSignupTrue = false; 
  String errorMessage = "";  

  signUp() async {
    // Validate if fields are empty
    if (fullName.text.isEmpty || email.text.isEmpty || usrName.text.isEmpty || password.text.isEmpty || confirmPassword.text.isEmpty) {
      setState(() {
        isSignupTrue = true;
        errorMessage = "All fields are required.";
      });
      return;
    }

    // Check if email already exists
    bool emailExists = await db.checkEmailExists(email.text);
    if (emailExists) {
      setState(() {
        isSignupTrue = true;
        errorMessage = "Email is already in use.";
      });
      return;
    }

    // Check if passwords match
    if (password.text != confirmPassword.text) {
      setState(() {
        isSignupTrue = true;
        errorMessage = "Passwords do not match.";
      });
      return;
    }

    // Proceed with user creation if checks pass
    var res = await db.createUser(Users(
      fullName: fullName.text,
      email: email.text,
      usrName: usrName.text,
      password: password.text,
    ));

    if (res > 0) {
      // Fetch the user details for the homepage
      Users? usrDetails = await db.getUser(usrName.text);

      // If result is successful, navigate to HomePage
      if (usrDetails != null) {
        if (!mounted) return;
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => HomePage(profile: usrDetails), // Navigate to HomePage
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
          child: SafeArea(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 20),
                  child: Text(
                    "Register New Account",
                    style: TextStyle(
                      color: primaryColor,
                      fontSize: 55,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                InputField(hint: "Full name", icon: Icons.person, controller: fullName),
                InputField(hint: "Email", icon: Icons.email, controller: email),
                InputField(hint: "Username", icon: Icons.account_circle, controller: usrName),
                InputField(hint: "Password", icon: Icons.lock, controller: password, passwordInvisible: true),
                InputField(hint: "Re-enter password", icon: Icons.lock, controller: confirmPassword, passwordInvisible: true),
                const SizedBox(height: 10),
                Button(label: "SIGN UP", press: () {
                  signUp();
                }),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text("Already have an account?", style: TextStyle(color: Colors.grey)),
                    TextButton(
                      onPressed: () {
                        Navigator.push(context, MaterialPageRoute(builder: (context) => const LoginScreen()));
                      },
                      child: const Text("LOGIN"),
                    ),
                  ],
                ),
                
                isSignupTrue
                    ? Text(errorMessage, style: TextStyle(color: Colors.red.shade900))
                    : const SizedBox(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
