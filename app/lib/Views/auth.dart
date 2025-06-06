import 'package:flutter/material.dart';
import 'package:app/Components/button.dart';
import 'package:app/Components/colors.dart';
import 'package:app/Views/login.dart';
import 'package:app/Views/signup.dart';

class AuthScreen extends StatelessWidget {
  const AuthScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
          child: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Let’s Get Started!",
                style: TextStyle(
                    fontSize: 35,
                    fontWeight: FontWeight.bold,
                    color: primaryColor),
              ),
              const Text(
                "Welcome to Your Robot Security Hub",
                style: TextStyle(color: Colors.grey),
              ),
              Expanded(child: Image.asset("assets/startup.jpg")),
              Button(label: "LOGIN", press: () {
                Navigator.push(context, MaterialPageRoute(builder: (context)=> const LoginScreen()));
              }),
              Button(label: "SIGN UP", press: () {
                Navigator.push(context, MaterialPageRoute(builder: (context)=> const SignupScreen()));
              }),
            ],
          ),
        ),
      )),
    );
  }
}
